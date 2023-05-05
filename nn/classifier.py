from nn.vae1 import Encoder, EncoderModule


from torch import nn
class Discriminator(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)