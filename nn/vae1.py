import torch
import torch.nn.functional as F


from torch import nn

class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)
        
class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()

        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # print('input', x.shape)
        x = self.convt(x)
        # print('ConvT', x.shape)

        x = self.bn(x)

        x = self.activation(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")


    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)
    
class VAE(nn.Module):

    def __init__(self, 
                pooling_kernel, 
                encoder_output_size, 
                color_channels ,
                num_classes = 0,
                rank: int = 0,
                *args, **kwargs) -> None:
        super().__init__()


        self.num_classes = num_classes

        self.n_latent_features = 64
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        self.device = torch.device(rank)
        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features + num_classes, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar
        
    def sample(self, label):
        
        c = F.one_hot(label, num_classes=self.num_classes)
        c = c.to(self.device)
        # assume latent features space ~ N(0, 1)
        z = torch.randn(len(c), self.n_latent_features).to(self.device)
        z = torch.cat([z,c], dim=1)
        z = self.fc3(z)
        # decode
        return self.decoder(z)
    
    def sample_class(self):
        label = torch.arange(0, self.num_classes)
        label = label.to(self.device)
        return self.sample(label)
    
    def forward(self, x, y):
        # Encoder
        h = self.encoder(x)

        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        c = F.one_hot(y, num_classes=self.num_classes)
        z = torch.cat([z, c], dim=1)
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):

        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        # BCE = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD