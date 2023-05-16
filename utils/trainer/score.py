

from .base import BaseTrainer



class ScoreTrainer(BaseTrainer):

    def __init__(self, model, dataloader, config, rank=0, *args, **kwargs) -> None:
        super().__init__(model, dataloader, config, rank, *args, **kwargs)

    def fit(self, x, y):
        logits = self.model(x)
        loss = self.model.loss(logits, y)  
        return loss

    def run(self):
        self.model.to(self.rank)
        for i, (x, y) in enumerate(self.data):
            x = x.to(self.rank)
            y = y.to(self.rank)
            loss = self.fit(x, y)
            loss.backward()
            self.optim.step()
            if(i + 1) % self.log_every == 0:
                self.logger.info(f'iter {i+1} loss {loss.item():.5f}')
                
        self.save(self.checkpoint_path + '.pt')
        self.logger.info(f"saving score model to {self.checkpoint_path}.pt")

