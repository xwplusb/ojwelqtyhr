from nn.vae1 import EncoderModule
import torch.nn.functional as F

from torch import nn
class Discriminator(nn.Module):

    def __init__(self, 
                pooling_kernel,
                encoder_output_size,
                color_channels,
                num_classes,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernel[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernel[1], kernel=3, pad=1)
        self.pool = nn.AvgPool2d(kernel_size=encoder_output_size)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        

        out = self.m3(self.m2(self.m1(self.bottle(x))))
        out = self.pool(out)
        out = self.head(out.squeeze(2,3))
        return out
    
    @staticmethod
    def loss(input, target):
        return F.cross_entropy(input, target)