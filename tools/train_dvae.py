import yaml
import argparse
import torch
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from model.discrete_vae import DiscreteVAE
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_color_texture_dataset import MnistVisualLanguageDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, crtierion, config):
    losses = []
    count = 0
    for data in tqdm(mnist_loader):
        # For vae we only need images
        im = data['image']
        im = im.float().to(device)
        optimizer.zero_grad()

        output, kl, log_qy = model(im)
        if config['train_params']['save_vae_training_image'] and count % 25 == 0:
            im_input = cv2.cvtColor((255 * (im.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0],
                                  cv2.COLOR_RGB2BGR)
            im_output = cv2.cvtColor((255 * (output.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0],
                                  cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/input.jpeg'.format(config['train_params']['task_name']), im_input)
            cv2.imwrite('{}/output.jpeg'.format(config['train_params']['task_name']), im_output)
        
        loss = (crtierion(output, im) + config['train_params']['kl_weight']*kl)/(1+config['train_params']['kl_weight'])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        count += 1
        
    print('Finished epoch: {} | Loss : {:.4f} '.
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
    
    
    #######################################
    # Create the model and dataset
    num_epochs = config['train_params']['num_epochs']
    model = DiscreteVAE(
        num_embeddings=config['model_params']['vae_num_embeddings'],
        embedding_dim=config['model_params']['vae_embedding_dim']
    )
    model.to(device)
    
    if os.path.exists('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])):
        print('Found checkpoint... Starting training from that')
        model.load_state_dict(torch.load('{}/{}'.format(config['train_params']['task_name'],
                                     config['train_params']['vae_ckpt_name'])))
    mnist = MnistVisualLanguageDataset('train', config['dataset_params'])
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'],
                              shuffle=True, num_workers=4)
    
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {
        'l1': torch.nn.SmoothL1Loss(beta=0.1),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])
    
    
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    
    best_loss = np.inf
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer, criterion, config)
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss from {:.4f} to {:.4f} .... Saving Model'.format(best_loss, mean_loss))
            torch.save(model.state_dict(), '{}/{}'.format(config['train_params']['task_name'],
                                                          config['train_params']['vae_ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement. Best Loss : {:.4f}'.format(best_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
