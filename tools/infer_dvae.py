import shutil
import yaml
import argparse
import torch
import os
import torchvision
from model.discrete_vae import DiscreteVAE
from dataset.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torchvision.utils import make_grid
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(args):
    r"""
    Method to infer discrete vae and get
    reconstructions
    :param args:
    :return:
    """
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    model = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    model.to(device)
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Inferring from that')
        model.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                                        config['train_params']['vae_ckpt_name']), map_location=device))
    else:
        print('No checkpoint found at {}/{}... Exiting'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name']))
        return
    model.eval()
    mnist = MnistVisualLanguageDataset('test', config['dataset_params'])
    
    # Generate reconstructions for 100 samples
    idxs = torch.randint(0, len(mnist) - 1, (25,))
    ims = torch.cat([mnist[idx]['image'][None, :] for idx in idxs]).float().to(device)
    output = model(ims)
    generated_im = output[0]
    
    # Dataset generates -1 to 1 we convert it to 0-1
    ims = (ims + 1) / 2
    generated_im = (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im])
    output = rearrange(out, 'b (c d) h w -> b (d) h (c w)', c=2, d=3)
    grid = make_grid(output, nrow=5)
    img = torchvision.transforms.ToPILImage()(grid.detach().cpu())
    img.save(os.path.join(config['train_params']['task_name'],
                          'dvae_reconstructions.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for discrete vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    inference(args)
