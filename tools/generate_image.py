import yaml
import argparse
import torch
import random
import os
import torchvision
import numpy as np
from einops import rearrange
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from model.dalle import DallE
from dataset.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    ######## Set the desired seed value #######
    # Ignoring the fixed seed value
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    # Create db to fetch the configuration values like vocab size (should do something better)
    mnist = MnistVisualLanguageDataset('train', config['dataset_params'])
    
    ###### Load Discrete VAE#####
    
    vae = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    vae.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Taking vae from that')
        vae.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                      config['train_params']['vae_ckpt_name']), map_location=device))
    else:
        print('No checkpoint found at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                                               config['train_params']['vae_ckpt_name']))
        print('Train vae first')
        return
    vae.eval()
    vae.requires_grad_(False)
    
    ##############################
    
    
    ########### Load DallE ##########
    model = DallE(vae=vae,
                  num_words=len(mnist.vocab_word_to_idx),
                  image_size=config['model_params']['dalle_image_size'],
                  max_text_len=mnist.max_token_len,
                  image_vocab_size=config['model_params']['vae_num_embeddings'],
                  gpt_config=config['gpt_config'])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['dalle_ckpt_name'])):
        print('Found checkpoint... Starting training from that')
        model.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                        config['train_params']['dalle_ckpt_name']), map_location=device))
    else:
        print('No checkpoint found for dalle at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                                                         config['train_params']['dalle_ckpt_name']))
    
        return
    #################################
    
    im_tokens_len = config['model_params']['dalle_image_size'] * config['model_params']['dalle_image_size']
    colors = ['red', 'blue', 'pink', 'green', 'cyan']
    textures = ['lego', 'stones', 'wool', 'cracker', 'peas']
    solids = ['orange', 'olive', 'purple']
    numbers = list(range(10))
    
    #### Genrate 10 random images ######
    vae_inputs = []
    fnames = []
    for _ in tqdm(range(10)):
        color = random.choice(colors)
        number = random.choice(numbers)

        ######## Set the desired seed value #######
        seed = np.random.randint(0, 1000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
            
        if random.random() < 0.1:
            solid = random.choice(solids)
            sent = ('generate image of {} in {} and a solid background of {}'
                    .format(number, color, solid).split(' '))
            fnames.append('{}_{}_{}.png'.format(number, color, solid))
        else:
            texture = random.choice(textures)
            sent = ('generate image of {} in {} and a texture background of {}'.
                    format(number, color, texture).split(' '))
            fnames.append('{}_{}_{}.png'.format(number, color, texture))
        sent = ['<bos>'] + sent + ['<sep>']
        text_tokens = torch.LongTensor([mnist.vocab_word_to_idx[word] for word in sent]).to(device).unsqueeze(0)
        random_im_tokens = torch.randint(0, config['model_params']['vae_num_embeddings'],
                                         (model.image_size * model.image_size,)).to(device)
        
        #### Generate pixels one by one #####
        im_tokens = torch.LongTensor([]).to(device)
        for tok_idx in range(im_tokens_len):
            logits, _, _ = model.gpt(im_tokens.unsqueeze(0), text_tokens)
            logits = logits[:, -1, :]
            
            # Ignore logits of all non-image tokens
            logits[:, :len(mnist.vocab_word_to_idx)] = -torch.finfo(logits.dtype).max
            
            # Get topk and sample from them
            val, ind = torch.topk(logits, 3)
            probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
            probs.scatter_(1, ind, val)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)[0].to(device)
            
            # Reduce predicted output by text vocab size to get vae token index
            sample -= model.num_words
            
            im_tokens = torch.cat((im_tokens, sample), dim=-1)
            random_im_tokens[:tok_idx + 1] = im_tokens
        
        
        vae_input = random_im_tokens.reshape((model.image_size, model.image_size))
        vae_inputs.append(vae_input.unsqueeze(0))
    
    # Pass predicted discrete sequence to vae
    vae_inputs = torch.cat(vae_inputs, dim=0)
    z = torch.nn.functional.one_hot(vae_inputs, num_classes=config['model_params']['vae_num_embeddings'])
    z = rearrange(z, 'b h w c -> b c h w').float()
    output = vae.decode_from_codebook_indices(z)
    output = (output + 1) / 2
    for idx in range((output.size(0))):
        img = torchvision.transforms.ToPILImage()(output[idx].detach().cpu())
        img.save(os.path.join(config['train_params']['task_name'],
                             fnames[idx]))
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generating outputs')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)