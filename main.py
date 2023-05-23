import yaml
import torch
import argparse

from pygad import GA
from torch.optim import Adam
from torch.utils.data import DataLoader

from nn.vae1 import VAE
from nn.classifier import Discriminator

from utils.data.sampler import yield_sample
from utils.data.dataloader import load_data
from utils.trainer import TeacherTrainer, ScoreTrainer, StudentTrainer
from utils.visulizer import save_img, save_plot
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_config.yaml")
    args = parser.parse_args()

    return args


def main(config):
    if 'train_teacher' in config['train']:
        data = load_data(**config['data'])
        model = VAE(**config['model'])
        dataloader = DataLoader(data, config['train']['batch_size'], shuffle=True, drop_last=True)
        trainer = TeacherTrainer(model, dataloader, config['train']['train_teacher'])
        trainer.run()
        teacher = trainer.model
    else:
        assert config['train']['teacher_path'], "teacher model path required"
        teacher = VAE(**config['model'])
        teacher.load_state_dict(torch.load(config['train']['teacher_path'])['model'])
        teacher.to('cuda')

    if 'train_score' in config['train']:
        score = Discriminator(**config['model'])
        # to sample training data once for all is not efficient
        # using a generator instead
        # TODO: find a better solution
        data = yield_sample(teacher,
                            config['model']['num_classes'],
                            config['train']['train_score']['batch_size'],
                            config['train']['train_score']['datasize'])
        score_trainer = ScoreTrainer(score, data, config['train']['train_score'])
        score_trainer.run()
    else:
        assert config['train']['score_path'], "score model path required"
        score = Discriminator(**config['model'])

        score.load_state_dict(torch.load(config['train']['score_path'])['model'])
        score.to('cuda')

    if 'train_students' in config['train']:
        student = VAE(**config['model'])
        targets = config['train']['student']['src']
        data = load_data(targets=targets, **config['data'])
        data_loader = DataLoader(data, config['train']['batch_size'])
        student_trainer = StudentTrainer(student, data_loader, config['train']['train_students'])
        student_trainer.run()
    else:
        assert config['train']['student_path'], "student model path is required"
        student = VAE(**config['model'])
        student.load_state_dict(torch.load(config['train']['student_path'])['model'])
        student.to('cuda')


    data_size = config['GA']['fit_func']['data_size']
    batch_size = config['GA']['fit_func']['batch_size']
    iter_limit = config['GA']['fit_func']['iter_limit']
    target_label = config['train']['student']['dst']
    target_label = torch.tensor(target_label).to('cuda')
    # target_label = torch.tensor([1,2,3,4,5,6]).to('cuda')
    grad_stats = [0 for i in range(config['GA']['instance']['num_generations'] + 1)]

    # avarager
    def fit_func(ga_instance, solution, solution_idx):

        idx = ga_instance.generations_completed
        with torch.no_grad():
            solution = torch.tensor(solution)
            beta = torch.softmax(solution, dim=0)

            beta = beta.to('cuda')
            head = torch.distributions.Categorical(probs=beta)

            csample = head.sample((data_size,))
            csample = target_label[csample]

        data = teacher.sample(csample)
        data = list(zip(data, csample))

        data_loader = DataLoader(data, batch_size)

        # print("examine new gene", solution_idx)
        # print("gene", beta)

        student_a = VAE(**config['model'])
        student_a.load_state_dict(student.state_dict())
        student_a.to('cuda')

        optim = Adam(student_a.parameters(), lr=config['GA']['fit_func']['lr'])

        iter_count = 0
        flag = True
        cnt_flag = True
        upper_cnt = iter_limit
        upper_bound = config['GA']['upper_bound']

        while flag:
            for i, (x, y) in enumerate(data_loader):
                x_, mu, log_var = student_a(x, y)
                loss = student_a.loss_function(x_, x, mu, log_var)
                optim.zero_grad()
                loss.backward()
                optim.step()
                iter_count += 1

                with torch.no_grad():

                    samples = student_a.sample(target_label)
                    logits = score(samples)
                    minus_grade = score.loss(logits, target_label)
                    # print(minus_grade, idx)
                    # if iter_count == 100:
                    #     exit(0)
                    if cnt_flag and minus_grade < upper_bound:
                        upper_cnt = iter_count
                        cnt_flag = False

                    if iter_count == iter_limit:
                        flag = False
                        grad_stats[idx] = minus_grade

                        class_samples = student_a.sample_class()
                        surfix = str(idx) + '_' + str(solution_idx) + '_' + str(iter_count)
                        save_img(class_samples, 4, 'output/images/meta_sample' + surfix + '.png')
                        break

        print(upper_cnt)

        # this is necessary ! 
        # or cuda out of memory
        del student_a
        torch.cuda.empty_cache()

        return 1 - upper_cnt / iter_limit

    init_samples = student.sample_class()
    save_img(init_samples, 4, 'output/images/init_sample.png')
    teacher.requires_grad_(False)
    score.requires_grad_(False)
    ga = GA(fitness_func=fit_func, **config['GA']['instance'])
    ga.run()
    save_plot(grad_stats, 'generation', 'best loss', 'output/images/fitness/grad.png')
    fitness_plot = ga.plot_fitness()
    fitness_plot.savefig(config['GA']['fitness_fig_path'])


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
