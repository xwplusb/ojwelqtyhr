import torch
from pygad import GA
from torch.utils.data import DataLoader

class Fitness():
    
    def __init__(self,teacher, student, score, config: dict):

        self.data_size = config['data_size']       
        self.batch_size = config['batch_size']

    def __call___(self, solution, solution_idx):
        solution = torch.tensor(solution)
        beta = torch.sofmax(solution)

        beta = beta.to('cuda')
        head = torch.distributions.Categorical(probs=beta)

        csample = head.sample((self.data_size))
        csample = self.target_label[csample]
        
        data = self.teacher.sample(csample)
        data = torch.cat([data, csample.unsqueeze(1)], dim=1)
        
        data_loader = DataLoader(data, batch_size=self.batch_size)
        print(1)
