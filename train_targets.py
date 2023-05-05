import yaml
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


from nn.test import VAE
from utils.data import MNIST_E, ToTensor

import matplotlib.pyplot as plt


def main(config):
    
    data = MNIST_E(config['data']['root'], transform=ToTensor(), targets=config['data']['targets'], download=True)
    data_loader = DataLoader(data, batch_size=config['data']['batch_size'], shuffle=True)



    vae = VAE(*config['vae']['init_params'])

    vae.to('cuda')
    vae_optim = Adam(vae.parameters(), lr=config['optim']['lr'])

    for epoch in range(config['train']['epochs']):

        for iteration, (x, y) in enumerate(data_loader):

            x = x.to('cuda')
            x = x.flatten(1)
            
            y = y.to('cuda')



            recon, mu, log_var = vae(x, y)

            loss = vae.loss(recon, x, mu, log_var, kl_weight=config['vae']['kl_weight'])

            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()

            if (iteration + 1) % config['train']['log_every'] == 0:
                print(loss.item())

        images = recon.view(-1,28,28)
        
        images = images[0]
        images = images.detach().cpu().numpy()
        plt.imsave(f"{epoch}.png", images)
      
    torch.save({
        "targets":config['data']['targets'],
        "encoder": vae.encoder.state_dict(),
        "decoder": vae.decoder.state_dict(),
    }, "vae" + ''.join(str(i) for i in config['data']['targets']) + '.pt')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)