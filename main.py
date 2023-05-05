import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from utils.data.dataloader import load_data
from utils.trainer.teacher import TeacherTrainer
from nn.vae1 import VAE


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_config.yaml")
    args = parser.parse_args()

    return args

def main(config):

    if config['train']['train_teacher']:
        data = load_data(**config['data'])
        model = VAE(**config['model'])
        dataloader = DataLoader(data, config['train']['batch_size'], drop_last=True)
        trainer = TeacherTrainer(model, dataloader, config['train']['train_teacher'])
        trainer.run()
        teacher = trainer.model
    else:
        assert config['train']['teacher_path'], "teacher model path required"
        teacher = VAE(**config['model'])
        teacher.load_state_dict(torch.load(config['train']['teacher_path']))


    if config['train']['train_score']:
        num_class = config['model']['num_classes']
        
        
    
    else:
        pass

    student = VAE(**config['model'])
    

    


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)