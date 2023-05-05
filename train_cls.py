import yaml
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam



from nn.classifier import Classifier
from utils.data import MNIST_E, ToTensor, MNIST
from nn.test import VAE



def main(config):
    
    init_params = [794, 256, 784, 2]
    vae = VAE(*init_params)
    vae.cuda()
    vae.decoder.load_state_dict(torch.load("output/checkpoints/vae0123456789.pt")['decoder'])
    vae.requires_grad_(False) 
    label = torch.randint(low=0, high=10, size=(10000,), device='cuda')
    
    sampled_data = vae.sample(label)
    sampled_data = torch.cat([sampled_data, label.unsqueeze(1)], dim=1)
    data_loader = DataLoader(sampled_data, batch_size=config['data']['batch_size'], shuffle=True)
    

    cls = Classifier
    cls.to('cuda')
    cls_optim = Adam(cls.parameters(), lr=config['optim']['lr'])

    loss = torch.nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(config['train']['epochs']):

        for iteration, xy in enumerate(data_loader):
            x, y = xy[:, :784], xy[:, -1]
            
            logits = cls(x)
            
            l = loss(logits, y.to(torch.int64))

            cls_optim.zero_grad()
            l.backward()
            cls_optim.step()
            if (iteration + 1) % config['train']['log_every'] == 0:
                print(l.item())
        
        with torch.no_grad():
            label = torch.randint(0, 10, (1000, ), device='cuda')
            sampled_data = vae.sample(label)
            logits = cls(sampled_data)
            pre_label = torch.softmax(logits, dim=1).argmax(dim=1)

            s = (pre_label == label).sum()
            print(s)


    torch.save({
        "targets":config['data']['targets'],
        "encoder": cls.state_dict(),
    }, "cls.pt")

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