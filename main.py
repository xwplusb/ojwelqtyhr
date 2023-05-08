import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from utils.data.dataloader import load_data
from utils.data.sampler import yield_sample
from utils.trainer import TeacherTrainer, ScoreTrainer, StudentTrainer
from nn.vae1 import VAE
from nn.classifier import Discriminator


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_config.yaml")
    args = parser.parse_args()

    return args

def main(config):

    # if 'train_teacher' in config['train']:
    #     data = load_data(**config['data'])
    #     model = VAE(**config['model'])
    #     dataloader = DataLoader(data, config['train']['batch_size'], drop_last=True)
    #     trainer = TeacherTrainer(model, dataloader, config['train']['train_teacher'])
    #     trainer.run()
    #     teacher = trainer.model
    # else:
    #     assert config['train']['teacher_path'], "teacher model path required"
    #     teacher = VAE(**config['model'])
    #     teacher.load_state_dict(torch.load(config['train']['teacher_path'])['model'])
    #     teacher.to('cuda')

    # if 'train_score' in config['train']:
    #     score = Discriminator(**config['model'])
    #     # to sample training data once for all is not efficient
    #     # using a generator instead
    #     # TODO: find a better solution :)
    #     data = yield_sample(teacher,
    #                          config['model']['num_classes'],
    #                          config['train']['score']['batch_size'],
    #                          config['train']['score']['datasize'])
    #     score_trainer = ScoreTrainer(score, data, config['train']['score'])
    #     score_trainer.run()
    # else:
    #     assert config['train']['score_path'], "score model path required"
    #     score = Discriminator(**config['model'])
    #     score.load_state_dict(torch.load(config['train']['score_path'])['model'])
    #     score.to('cuda')
    


        
    if 'train_students' in config['train']:
        student = VAE(**config['model'])
        targets = config['train']['student']['src']
        data = load_data(targets=targets, **config['data'])
        data_loader = DataLoader(data, config['train']['batch_size'], drop_last=True)
        student_trainer = StudentTrainer(student, data_loader, config['train']['train_students'])
        student_trainer.run()
    
    else:
        assert config['train']['student_path'], "student model path is required"
        student = VAE(**config['model'])
        student.load_state_dict(torch.load(config['train']['student_path'])['model'])
        student.to('cuda')
    

    


if __name__ == '__main__':

    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)