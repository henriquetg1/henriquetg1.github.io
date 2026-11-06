"""A variational autoencoder for flattened MNIST, written by hand.

PyTorch provides the layers, autograd and Adam. The VAE-specific parts are all
here: the reparameterization, both terms of the loss, and sampling from the prior.
"""

import torch
import torch.nn.functional as F
from torch import nn

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20


class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256, heads 256 -> d, decoder d -> 256 -> 512 -> 784."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps. The noise sits outside the path from mu and sigma,
        # so gradients can flow through both of them.
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n):
        # z ~ N(0, I), the prior, pushed through the decoder
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


def loss_terms(x, x_hat, mu, logvar):
    """Return both loss terms in nats per image: summed over pixels, averaged over images."""
    batch_size = x.shape[0]
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
    # Closed-form KL between N(mu, sigma^2) and N(0, I), summed over latent dimensions
    kl = 0.5 * torch.sum(mu**2 + logvar.exp() - 1 - logvar) / batch_size
    return reconstruction, kl


@torch.no_grad()
def evaluate(model, x):
    """Average both loss terms over a dataset, with z sampled as in training."""
    totals = torch.zeros(2)
    for batch in x.split(BATCH_SIZE):
        x_hat, mu, logvar = model(batch)
        reconstruction, kl = loss_terms(batch, x_hat, mu, logvar)
        totals += torch.stack((reconstruction, kl)) * batch.shape[0]
    return (totals / x.shape[0]).tolist()


def train(model, x_train, x_val, epochs=EPOCHS):
    """Train with Adam (beta = 1) and log both terms on train and validation each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}

    for _ in range(epochs):
        model.train()
        totals = torch.zeros(2)
        for indices in torch.randperm(x_train.shape[0]).split(BATCH_SIZE):
            batch = x_train[indices]
            x_hat, mu, logvar = model(batch)
            reconstruction, kl = loss_terms(batch, x_hat, mu, logvar)
            loss = reconstruction + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += torch.stack((reconstruction, kl)).detach() * batch.shape[0]

        # Training numbers are the running average over the epoch, as the statement asks
        train_rec, train_kl = (totals / x_train.shape[0]).tolist()
        model.eval()
        val_rec, val_kl = evaluate(model, x_val)

        history["train_rec"].append(train_rec)
        history["train_kl"].append(train_kl)
        history["val_rec"].append(val_rec)
        history["val_kl"].append(val_kl)

    return history
