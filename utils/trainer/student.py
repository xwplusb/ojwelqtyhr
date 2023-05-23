from .base import BaseTrainer
from torch import no_grad
from utils.visulizer import save_img
from torch.optim.lr_scheduler import StepLR

class StudentTrainer(BaseTrainer):

    def __init__(self, model, dataloader, config, rank=0, *args, **kwargs) -> None:
        super().__init__(model, dataloader, config, rank, *args, **kwargs)
        self.sample_path = config['sample_path']
        self.save_every = config['save_every']
        self.lr_schr = StepLR(self.optim, 10, 0.1)
        
    def fit(self, x, y):
        x_, mu, logvar = self.model(x, y)
        loss = self.model.loss_function(x_, x, mu, logvar)
        return loss, x_
    
    def run(self):

        self.model = self.model.to(self.rank)

        for epoch in range(self.epoch):
            for i, (x, y) in enumerate(self.data):
                x = x.to(self.rank)
                y = y.to(self.rank)
                loss, output = self.fit(x, y)            
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if (i + 1) % self.log_every  == 0:
                    self.logger.info(f'iter {i+1} loss {loss.item():.5f}')
            self.lr_schr.step()
            if (epoch + 1) % self.save_every == 0:
                    self.save(self.checkpoint_path + str(epoch) + '.pt')
                    self.logger.info(f'saving checkpoint to {self.checkpoint_path + str(epoch) + ".pt"}')
                    with no_grad():
                        sample = self.model.sample_class()

                        save_img(sample, 4, self.sample_path + str(epoch) + '.png')
                        self.logger.info(f'saveing sampled images to {self.sample_path + str(epoch) + ".png"}')
