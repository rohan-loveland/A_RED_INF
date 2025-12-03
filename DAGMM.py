import math
import torch
import torch.nn as nn

class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=16, n_components=4):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.Tanh(),
            nn.Linear(1024, 512), nn.Tanh(),
            nn.Linear(512, 128), nn.Tanh(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.Tanh(),
            nn.Linear(128, 512), nn.Tanh(),
            nn.Linear(512, 1024), nn.Tanh(),
            nn.Linear(1024, input_dim), nn.Tanh()
        )

        self.gmm_phi    = nn.Parameter(torch.ones(n_components) / n_components)
        self.gmm_mu     = nn.Parameter(torch.randn(n_components, latent_dim))
        self.gmm_logvar = nn.Parameter(torch.log(torch.ones(n_components, latent_dim)))

        self.energy_net = nn.Sequential(
            nn.Linear(latent_dim + n_components, 64), nn.Tanh(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.Tanh(), nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        phi = torch.softmax(self.gmm_phi, dim=0)
        mu, logvar = self.gmm_mu, self.gmm_logvar
        z_exp = z.unsqueeze(1)
        diff = z_exp - mu.unsqueeze(0)

        log_prob = -0.5 * (
            self.latent_dim * math.log(2 * math.pi) +
            logvar.unsqueeze(0) +
            diff.pow(2) / torch.exp(logvar.unsqueeze(0))
        )
        log_prob = log_prob.sum(dim=2) + torch.log(phi + 1e-10)
        gamma = torch.softmax(log_prob, dim=1)

        energy = self.energy_net(torch.cat([z, gamma], dim=1)).squeeze(1)
        return z, x_rec, gamma, energy
