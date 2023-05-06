import logging

from torch import save, load
from torch.optim import Adam


class BaseTrainer:

    def __init__(self, model, dataloader, config, rank=0, *args, **kwargs) -> None:
        
        self.epoch = config['epoch']
        self.model = model
        self.data = dataloader
        self.rank = rank
        self.optim = Adam(params=model.parameters(), lr=config['lr'])

        self.checkpoint_path = config['checkpoint_path']
        self.logger = self.get_logger(config['log'])
        self.log_every = config['log']['log_every']
    
    def run(self):
        raise NotImplementedError

    def save(self, path):
        save({
            'model': self.model.state_dict(),
        }, path)
    
    def load(self, path):
        self.model.load_state_dict(load(path)['model'])

    @staticmethod
    def get_logger(config):

        logging.basicConfig(filename=config['path'], level=logging.INFO)
        logger = logging.getLogger(name=config['path'])
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(config['path'])

        formatter = logging.Formatter('%(asctime)s-%(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def fit(self, x, y):
        raise NotImplementedError
    
class StudentTrainer:

    def __init__(self, teacher, student, discriminator, config, *args, **kwargs) -> None:
        
        pass

    def run(self):
        
        for i in range(self.epoch):
            pass 
    


# class BaseClassifier():

#     def __init__(self) -> None:
         