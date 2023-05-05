import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt


from nn.test import VAE
from nn.classifier import Classifier

import pygad


init_params = [794, 256, 784, 2]
vae0123456789 = VAE(*init_params)
vae0123 = VAE(*init_params)
cls = Classifier

vae0123456789.decoder.load_state_dict(torch.load("output/checkpoints/vae0123456789.pt")['decoder'])
vae0123456789.requires_grad_(False)

cls.load_state_dict(torch.load("output/checkpoints/cls.pt")['encoder'])
cls.requires_grad_(False)

vae0123456789.to('cuda')
cls.to('cuda')



sample_size = 256


meta_loss = torch.nn.CrossEntropyLoss(reduction='sum')
iter_counts = []


def fitness_func(ga_instance, solution, solution_idx):


        solution = torch.tensor(solution)
        beta = torch.softmax(solution, dim=0)


        # generating dataset
        #
        # sample label index according to beta
        # index labels
        # sample mnist 784 dim images from labels
        # create dataloader from sampled images together with label
        beta = beta.to('cuda')
        head = torch.distributions.Categorical(probs=beta)
        sample_labels = torch.tensor([4, 7, 5, 6]).to('cuda')
        csample = head.sample((4096,))
        csample = sample_labels[csample]
        meta_data = vae0123456789.sample(csample)
        meta_data = torch.cat([meta_data, csample.unsqueeze(1)], dim=1)
        meta_loader = DataLoader(meta_data, batch_size=32)
        
        flag = True
        iter_count = 0
        loss_upper_bound = 35 

        vae0123.decoder.load_state_dict(torch.load("output/checkpoints/vae0123.pt")['decoder'])
        vae0123.encoder.load_state_dict(torch.load("output/checkpoints/vae0123.pt")['encoder'])
        vae0123.to('cuda')
        vae_optim = Adam(vae0123.parameters(), lr=0.001)

        print("start new gene", solution_idx)
        print("gene:", beta)
        while flag and iter_count < 1000:

            min_meta_ls = 1000

            for i, xy in enumerate(meta_loader):
                x, y = xy[:, :784], xy[:, -1]
                recon, mu, log_var = vae0123(x, y.to(torch.int64))
                loss = vae0123.loss(recon, x, mu, log_var, kl_weight=0.005)

                vae_optim.zero_grad()
                loss.backward()
                vae_optim.step()
                iter_count += 1

                with torch.no_grad():
                    choices = torch.randint(sample_labels.size(-1), (sample_size ,))
                    sample_labels = sample_labels[choices]
                    samples = vae0123.sample(sample_labels)
                    sample_logits = cls(samples)
                    meta_l = meta_loss(sample_logits, sample_labels)
                    min_meta_ls = min(min_meta_ls, meta_l)
                    if min_meta_ls < loss_upper_bound or iter_count == 1000:
                        flag = False
                        break
            
        print(iter_count)

        # iter_counts.append(iter_count)
        return 1 - iter_count/1000

fitness_function = fitness_func

num_generations = 32
num_parents_mating = 8

sol_per_pop = 10
num_genes = 4
init_range_low = 0
init_range_high = 1

parent_selection_type = "sss"
keep_parents = 4

crossover_type = "uniform"

mutation_type = "scramble" 
mutation_percent_genes = 1




ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                    #    mutation_percent_genes=mutation_percent_genes,
                        mutation_num_genes=2,
                       )
ga_instance.max_num_iteration = 32
ga_instance.run()


fitness_plot = ga_instance.plot_fitness()
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print(solution_fitness)
fitness_plot.savefig('fitness_plot.png')
