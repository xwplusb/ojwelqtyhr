import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt

from io import BytesIO

from nn.vae1 import VAE
from torch.optim import Adam
from nn.classifier import Discriminator
from torchvision.utils import save_image
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_config.yaml")
    ret_args = parser.parse_args()

    return ret_args


def main(config):
    pt_stat = os.stat(config['train']['teacher_path'])
    teacher_volum = pt_stat.st_size / 2 ** 10

    score = Discriminator(**config['model'])
    score.load_state_dict(torch.load(config['train']['score_path'])['model'])
    score.to('cuda')

    teacher = VAE(**config['model'])
    teacher.load_state_dict(torch.load(config['train']['teacher_path'])['model'])
    teacher.to('cuda')

    # raw_pt = VAE(**config['model'])
    # raw_pt.load_state_dict(torch.load(config['train']['teacher_path'])['model'])
    # raw_pt.to('cuda')

    student = VAE(**config['model'])

    target_label = config['train']['student']['dst']
    target_label = torch.tensor(target_label).to('cuda')

    accumulate_size = 1000

    iter_batch_size = 10
    iter_count = 10

    grades = []
    volum = []

    for target_size in range(100, 1000, 100):

        indices = torch.randint(0, target_label.shape[0], (target_size,))
        labels = target_label[indices]
        images = teacher.sample(labels).detach()

        images += torch.randn_like(images) * 1.0

        image_size = 0
        # CAUTION: using bytesIO to count the file size without actual saving
        # Try find better solution!
        for img in images:
            IMAGE_FILE = BytesIO()
            save_image(img, IMAGE_FILE, format='jpeg')
            image_size += IMAGE_FILE.getbuffer().nbytes
            IMAGE_FILE.close()

        volum.append(image_size / 2 ** 10)
        data = list(zip(images, labels))
        data_loader = DataLoader(data, iter_batch_size)

        student.load_state_dict(torch.load(config['train']['student_path'])['model'])
        optim = Adam(student.parameters(), lr=2e-4)
        student.to('cuda')

        for _ in range(iter_count):
            for i, (x, y) in enumerate(data_loader):
                x_, mu, log_var = student(x, y)
                loss = student.loss_function(x_, x, mu, log_var)
                optim.zero_grad()
                loss.backward()
                optim.step()

        with torch.no_grad():
            samples = student.sample(target_label)
            logits = score(samples)
            grade = score.loss(logits, target_label)
            grades.append(grade.item())
            print(grades[-1])

    # tgrades = []
    
    # for std in torch.linspace(0, 4, 40):
    #     with torch.no_grad():
    #         for p in teacher.parameters():
    #             p += torch.randn_like(p) * std
    #         teacher_grade = 0
    #         samples = teacher.sample(target_label)
    #         logits = score(samples)
    #         teacher_grade = score.loss(logits, target_label).item()
    #         tgrades.append(teacher_grade)
    # plt.plot(torch.linspace(0, 4, 40), tgrades)
    # plt.xlabel('std')
    # plt.ylabel('grade')
    # plt.title('tx pt file through noise channel')
    # plt.savefig('Images/标准差vs图片质量.png')
    # plt.show()
    # exit()


    # plt.scatter(teacher_volum, teacher_grade)
    plt.plot(volum, grades)
    plt.xlabel('KBytes')
    plt.ylabel('minus score')
    plt.title('rx image with std=1.0 noise; DataSize VS Score')
    plt.savefig('Images/含噪数据量vs图片质量1.png')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
