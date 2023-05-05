import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt


from nn.test import VAE
from nn.classifier import Classifier

import pygad

def fitness_func(ga_instance, solution, solution_idx):
    print(ga_instance, solution, solution_idx)

fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = 4

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


init_params = [794, 256, 784, 2]
vae0247 = VAE(*init_params)
vae0489 = VAE(*init_params)
cls = Classifier

vae0247.decoder.load_state_dict(torch.load("output/checkpoints/vae0247.pt")['decoder'])
vae0247.requires_grad_(False)

cls.load_state_dict(torch.load("output/checkpoints/cls.pt")['encoder'])
cls.requires_grad_(False)

vae0247.to('cuda')
cls.to('cuda')


sample_labels = torch.tensor([0, 2, 4, 7]).to('cuda')

sample_size = 256
choices = torch.randint(sample_labels.size(0), (sample_size ,))
sample_labels = sample_labels[choices]


vae_optim = Adam(vae0489.parameters(), lr=0.001)

meta_loss = torch.nn.CrossEntropyLoss(reduction='sum')
loss_upper_bound = 11
iter_counts = []


for epoch in range(100):


    beta = torch.randint(0, 100, size=(4,))
    beta = beta / beta.sum()
    beta = beta.to('cuda')
    head = torch.distributions.Categorical(probs=beta)


    csample = head.sample((2048,))
    csample = sample_labels[csample]

    meta_data = vae0247.sample(csample)
    meta_data = torch.cat([meta_data, csample.unsqueeze(1)], dim=1)
    meta_loader = DataLoader(meta_data, batch_size=16)
    
    iter_count = 0
    loss_upper_bound = 1e10

    vae0489.decoder.load_state_dict(torch.load("output/checkpoints/vae0489.pt")['decoder'])
    vae0489.encoder.load_state_dict(torch.load("output/checkpoints/vae0489.pt")['encoder'])
    vae0489.to('cuda')

    losses = []

    while loss_upper_bound > 100 and iter_count < 3000:

        meta_ls = []
        for i, xy in enumerate(meta_loader):
            x, y = xy[:, :784], xy[:, -1]
            recon, mu, log_var = vae0489(x, y.to(torch.int64))
            loss = vae0489.loss(recon, x, mu, log_var, kl_weight=0.005)

            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()

            iter_count += 1
            samples = vae0489.sample(sample_labels)        
            sample_logits = cls(samples)
            meta_l = meta_loss(sample_logits, sample_labels)
            meta_ls.append(meta_l)
        loss_upper_bound = torch.mean(torch.tensor(meta_ls))

    iter_counts.append(iter_count)

xais = torch.arange(100)
yais = torch.tensor(iter_counts)
yais = yais.detach().cpu().numpy()

plt.plot(xais, yais)
plt.xlabel('epoch')
plt.ylabel('iter')
plt.savefig('plot1.png')