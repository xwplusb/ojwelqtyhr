import torch
from torch import nn



class VAE(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, mu_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        self.encoder = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU()
        )

        self.estimate_mu = nn.Linear(mid_dim, mu_dim)
        self.estimate_var = nn.Linear(mid_dim, mu_dim)

        self.decoder = nn.Sequential(
            nn.Linear(mu_dim+10, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid()
        )
    
    def sample(self, labels):

        z = torch.randn(labels.size()[0], 2, device=labels.device)
        c = torch.zeros(labels.size(0), 10, device=labels.device)
        c.scatter_(1, labels.unsqueeze(1), 1)

        z = torch.cat([z, c], dim=1)
        return self.decoder(z)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    @staticmethod
    def loss(recon, x, mu, log_var, kl_weight, reduction='sum'):
        bce = torch.nn.functional.binary_cross_entropy(recon, x, reduction=reduction)

        kld = kl_weight * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return (bce - kld) / x.size(0)
    
    def forward(self, x, y):

        c = torch.zeros(y.size(0), 10, device=x.device)
        c.scatter_(1, y.unsqueeze(1), 1)
        
        x = torch.cat([x, c], dim=1)

        x = self.encoder(x)
        mu, log_var = self.estimate_mu(x), self.estimate_var(x)

        z = self.reparameterize(mu, log_var)


        z = torch.cat((z, c), dim=1)
        recon = self.decoder(z)

        return recon, mu, log_var
    



class VAVAE(nn.Module):


    def __init__(self, in_dim, mid_dim, out_dim, mu_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



        self.encoder1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU()
        )

        self.estimate_mu1 = nn.Linear(mid_dim, mid_dim)
        self.estimate_var1 = nn.Linear(mid_dim, mid_dim)


        self.estimate_mu2 = nn.Linear(mid_dim*2, mu_dim)
        self.estimate_var2 = nn.Linear(mid_dim*2, mu_dim)

        self.decoder = nn.Sequential(
            nn.Linear(mu_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        
        x1 = self.encoder1(x)
        mu1, var1 = self.estimate_mu1(x1), self.estimate_var2(x1)
        z1 = self.reparameterize(mu1, var1)
        del x1

        x2 = self.encoder(x)
        x2 = torch.cat([x2, z1], dim=1)

        mu2, var2 = self.estimate_mu2(x2), self.estimate_var2(x2)
        z2 = self.reparameterize(mu2, var2)

        recon = self.decoder(z2)

        return recon, mu1, var1, mu2, var2

    @staticmethod
    def loss(recon, x, mu1, log_var1, kl_weight1, mu2, log_var2, kl_weight2, reduction='sum'):
        bce = torch.nn.functional.binary_cross_entropy(recon, x, reduction=reduction)

        kld1 = kl_weight1 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp())
        kld2 = kl_weight2 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp())

        return (bce - kld1 - kld2) / x.size(0)


    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)
