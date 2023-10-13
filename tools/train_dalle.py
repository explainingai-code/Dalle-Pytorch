import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from model.dalle import DallE
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt_counts = 0


def train_for_one_epoch(epoch_idx, model, loader, optimizer, config):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: Dalle model
    :param mnist_loader: Data loder
    :param optimizer: optimzier to be used taken from config
    :param crtierion: For computing the loss
    :param config: configuration for the current run
    :return:
    """
    losses = []
    for data in tqdm(loader):
        im = data['image']
        text_tokens = data['text_tokens']
        im = im.float().to(device)
        text = text_tokens.long().to(device)
        optimizer.zero_grad()
        
        _, loss_text, loss_image = model(im, text)
        loss = (loss_text*1 + loss_image*config['train_params']['dalle_image_loss']) / (1+config['train_params']['dalle_image_loss'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Modelling Loss : {:.4f} '.
          format(epoch_idx + 1,
                 np.mean(losses)))
    return np.mean(losses)


def train(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    #######################################
    
    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    ######## Create the model and dataset ##########
    num_epochs = config['train_params']['num_epochs_dalle']
    mnist = MnistVisualLanguageDataset('train', config['dataset_params'])
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['dalle_batch_size'],
                              shuffle=True, num_workers=4)
    vae = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    vae.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Taking vae from that')
        vae.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                        config['train_params']['vae_ckpt_name']),map_location=device))
    else:
        print('No checkpoint found at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                                               config['train_params']['vae_ckpt_name']))
        print('Train vae first')
        return
    vae.eval()
    vae.requires_grad_(False)
    
    
    model = DallE(vae=vae,
                  num_words=len(mnist.vocab_word_to_idx),
                  image_size=config['model_params']['dalle_image_size'],
                  max_text_len=mnist.max_token_len,
                  image_vocab_size=config['model_params']['vae_num_embeddings'],
                  gpt_config=config['gpt_config'])
    model.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['dalle_ckpt_name'])):
        print('Found checkpoint... Starting training from that')
        model.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['dalle_ckpt_name']),map_location=device))
        
    ####### Training Parameters ############
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)

    best_loss = np.inf
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, config)
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss from {:.4f} to {:.4f} .... Saving Model'.format(best_loss, mean_loss))
            torch.save(model.state_dict(), '{}/{}'.format(config['train_params']['task_name'],
                                                          config['train_params']['dalle_ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement. Best Loss : {:.4f}'.format(best_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for dalle training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
