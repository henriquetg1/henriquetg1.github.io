"""Runs Exercises 1 and 2: data, training, Figures 1-6 and every reported number.

Usage, from the repository root: python docs/exercises/vae/code/exercises.py
MNIST is downloaded into ./data on the first run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from scipy.stats import norm
from torch.utils.data import random_split
from torchvision import datasets, transforms

from vae import EPOCHS, VAE, evaluate, loss_terms, train

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

SEED = 42
INTERPOLATION_PAIR = (3, 2)  # test indices: the first 0 and the first 1

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECONDARY_INK = "#0b0b0b", "#52514e"
# Eight palette hues, then two repeated with another marker, so ten digits stay distinct
DIGIT_STYLES = [
    (BLUE, "o"), (ORANGE, "o"), ("#1baf7a", "o"), ("#eda100", "o"), ("#e87ba4", "o"),
    ("#008300", "o"), ("#6250d6", "o"), ("#e34948", "o"), (BLUE, "^"), (ORANGE, "^"),
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": SECONDARY_INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.labelcolor": SECONDARY_INK,
        "ytick.labelcolor": SECONDARY_INK,
        "xtick.color": "#c3c2b7",
        "ytick.color": "#c3c2b7",
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def load_data():
    """The statement's split, then all pixels as tensors so that an epoch takes seconds."""
    train_full = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(
        train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED)
    )

    # Same values as ToTensor: bytes divided by 255, flattened into 784-vectors
    pixels = train_full.data.float().flatten(1) / 255
    assert torch.equal(pixels[0], train_full[0][0].flatten())
    x_test = test.data.float().flatten(1) / 255
    return pixels[train_set.indices], pixels[val_set.indices], x_test, test.targets


def tile(images, rows, cols, size=28, gap=2):
    """Lay flattened images out on one canvas, with a faint line between the cells."""
    images = images.reshape(-1, size, size).detach().numpy()
    step = size + gap
    canvas = np.full((rows * step - gap, cols * step - gap), 0.12)
    for i, image in enumerate(images):
        row, col = divmod(i, cols)
        canvas[row * step : row * step + size, col * step : col * step + size] = image
    return canvas


def show_grid(images, rows, cols, title, xlabel, ylabel, name, xticks=(), yticks=(), scale=0.62):
    """Show a tile of images with a title, axis labels and optional tick labels."""
    fig, ax = plt.subplots(figsize=(cols * scale, rows * scale + 0.5))
    ax.imshow(tile(images, rows, cols), cmap="gray_r", vmin=0, vmax=1)
    centers = lambda n: [30 * i + 13.5 for i in range(n)]
    ax.set_xticks(centers(cols) if len(xticks) else [], xticks)
    ax.set_yticks(centers(rows) if len(yticks) else [], yticks)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def train_model(latent_dim, x_train, x_val):
    torch.manual_seed(SEED)  # before building each model, as the statement requires
    model = VAE(latent_dim)
    history = train(model, x_train, x_val)
    model.eval()
    return model, history


def plot_history(history):
    epochs = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    panels = (
        (axes[0], "rec", "Reconstruction (nats per image)", "Reconstruction term"),
        (axes[1], "kl", "KL (nats per image)", "KL term"),
    )
    for ax, key, ylabel, title in panels:
        ax.plot(epochs, history[f"train_{key}"], color=BLUE, label="Training")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend()
    fig.suptitle("Figure 1: both loss terms of the VAE with d = 16", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1.png")
    plt.close(fig)


@torch.no_grad()
def exercise_1(x_train, x_val, x_test, y_test):
    print("=== Exercise 1 ===")
    print(f"Train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}")
    with torch.enable_grad():
        model, history = train_model(16, x_train, x_val)

    print("Epoch | train rec | train KL | val rec | val KL")
    for i in range(EPOCHS):
        print(
            f"{i + 1:5d} | {history['train_rec'][i]:9.2f} | {history['train_kl'][i]:8.2f} | "
            f"{history['val_rec'][i]:7.2f} | {history['val_kl'][i]:6.2f}"
        )
    plot_history(history)

    test_rec, test_kl = evaluate(model, x_test)
    print(f"Test reconstruction {test_rec:.2f}, KL {test_kl:.2f}, "
          f"negative ELBO {test_rec + test_kl:.2f}")

    # Figure 2: first test image of each digit, reconstructed by decoding mu
    first = [int((y_test == digit).nonzero()[0]) for digit in range(10)]
    print(f"First test index of digits 0-9: {first}")
    mu, logvar = model.encode(x_test[first])
    show_grid(
        torch.cat((x_test[first], model.decode(mu))), 2, 10,
        "Figure 2: test images and their reconstructions from $\\mu$",
        "Digit", "Row", "fig2.png",
        xticks=range(10), yticks=["Original", "Reconstruction"], scale=0.8,
    )

    # Figure 3: samples from the prior
    torch.manual_seed(0)
    show_grid(
        model.sample(64), 8, 8,
        "Figure 3: images decoded from $z \\sim N(0, I)$",
        "Sample column", "Sample row", "fig3.png", scale=0.8,
    )

    # Numbers for item E. Decoding mu against decoding a sampled z, on the whole test set:
    mu, logvar = model.encode(x_test)
    from_mu, _ = loss_terms(x_test, model.decode(mu), mu, logvar)
    print(f"Test reconstruction decoding mu: {from_mu:.2f} (decoding a sampled z: {test_rec:.2f})")

    # Do prior samples land where the encoded images are? Compare in the active dimensions
    # only: the decoder ignores the others, so noise there would inflate every distance.
    kl_per_dim = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).mean(dim=0)
    active = kl_per_dim > 0.1
    print(f"KL per latent dimension: {kl_per_dim.numpy().round(2)}")
    print(f"Active dimensions (KL > 0.1): {int(active.sum())} of 16")
    tree = cKDTree(model.encode(x_train)[0][:, active].numpy())
    groups = (
        ("test mu", mu),
        ("z sampled from test posteriors", model.reparameterize(mu, logvar)),
        ("z sampled from the prior", torch.randn(10_000, 16)),
    )
    for name, points in groups:
        distance = tree.query(points[:, active].numpy(), workers=-1)[0]
        print(f"Distance to the nearest encoded training image, {name}: "
              f"median {np.median(distance):.2f}, 90th percentile {np.quantile(distance, 0.9):.2f}")
    return model, history, test_rec, test_kl


def plot_encoded(mu, y_test, grid):
    fig, ax = plt.subplots(figsize=(8, 7))
    for digit, (color, marker) in enumerate(DIGIT_STYLES):
        points = mu[y_test == digit]
        ax.scatter(points[:, 0], points[:, 1], s=6, c=color, marker=marker,
                   alpha=0.7, linewidths=0, label=str(digit))
    for digit in range(10):  # direct labels, so identity never rests on color alone
        cx, cy = np.median(mu[y_test == digit], axis=0)
        text = ax.text(cx, cy, str(digit), ha="center", va="center", fontsize=13,
                       fontweight="bold", color=INK)
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
    ax.add_patch(Rectangle((grid[0], grid[0]), grid[-1] - grid[0], grid[-1] - grid[0],
                           fill=False, edgecolor=SECONDARY_INK, linewidth=1, label="Figure 5 area"))
    ax.set(
        title="Figure 4: $\\mu$ of all 10 000 test images, d = 2",
        xlabel="$\\mu_1$",
        ylabel="$\\mu_2$",
    )
    ax.legend(title="Digit", loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=3)
    fig.savefig(FIGURES / "fig4.png")
    plt.close(fig)


@torch.no_grad()
def exercise_2(model_16, rec_16, kl_16, x_train, x_val, x_test, y_test):
    print("\n=== Exercise 2 ===")
    with torch.enable_grad():
        model, history = train_model(2, x_train, x_val)
    test_rec, test_kl = evaluate(model, x_test)
    print(f"d = 2:  test reconstruction {test_rec:.2f}, KL {test_kl:.2f}, "
          f"negative ELBO {test_rec + test_kl:.2f}")
    print(f"d = 16: test reconstruction {rec_16:.2f}, KL {kl_16:.2f}, "
          f"negative ELBO {rec_16 + kl_16:.2f}")
    print(f"ln(10), the nats needed just to name a digit: {np.log(10):.2f}")

    # Figure 5: decode a 15 x 15 grid at evenly spaced quantiles of N(0, 1)
    grid = norm.ppf(np.linspace(0.05, 0.95, 15))
    print(f"Grid values: {grid.round(2)}")
    z1, z2 = np.meshgrid(grid, grid[::-1])  # z2 grows upward, the same layout as Figure 4
    z = torch.tensor(np.c_[z1.ravel(), z2.ravel()], dtype=torch.float32)
    ticks = [f"{value:.2f}" for value in grid]
    show_grid(
        model.decode(z), 15, 15,
        "Figure 5: the decoded plane, on the $\\Phi^{-1}$ grid",
        "$z_1$", "$z_2$", "fig5.png",
        xticks=ticks, yticks=ticks[::-1],
    )

    # Figure 4, with the area that Figure 5 covers marked on it
    mu = model.encode(x_test)[0].numpy()
    labels = y_test.numpy()
    plot_encoded(mu, labels, grid)
    inside = np.all(np.abs(mu) <= grid[-1], axis=1).mean()
    print(f"Test points inside the Figure 5 area: {inside:.1%}")

    # Which digits overlap? How often a point's nearest neighbour is another digit.
    neighbour = cKDTree(mu).query(mu, k=2, workers=-1)[1][:, 1]
    mixing = np.zeros((10, 10), dtype=int)
    np.add.at(mixing, (labels, labels[neighbour]), 1)
    cross = mixing + mixing.T
    np.fill_diagonal(cross, 0)
    pairs = sorted(
        ((int(cross[a, b]), a, b) for a in range(10) for b in range(a + 1, 10)), reverse=True
    )
    print(f"Points whose nearest neighbour is another digit: {1 - np.trace(mixing) / len(mu):.1%}")
    print("Top overlapping pairs (nearest-neighbour links):")
    for count, a, b in pairs[:6]:
        print(f"  {a} and {b}: {count}")
    print("Median mu per digit:")
    for digit in range(10):
        print(f"  {digit}: {np.median(mu[labels == digit], axis=0).round(2)}")

    # Figure 6: interpolate in pixel space and in the latent space of the d = 16 model
    a, b = INTERPOLATION_PAIR
    print(f"Interpolated test indices: {a} (digit {int(y_test[a])}) and {b} (digit {int(y_test[b])})")
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    x_a, x_b = x_test[a], x_test[b]
    pixel_row = (1 - t) * x_a + t * x_b
    mu_a, mu_b = model_16.encode(x_a.unsqueeze(0))[0], model_16.encode(x_b.unsqueeze(0))[0]
    latent_row = model_16.decode((1 - t) * mu_a + t * mu_b)
    show_grid(
        torch.cat((pixel_row, latent_row)), 2, 10,
        f"Figure 6: from test image {a} to test image {b}",
        "$t$", "Interpolation space", "fig6.png",
        xticks=[f"{value:.2f}" for value in t.squeeze(1).tolist()],
        yticks=["Pixel", "Latent"], scale=0.85,
    )
    return test_rec, test_kl


def main():
    torch.set_num_threads(1)  # one thread keeps the floating-point results repeatable
    x_train, x_val, x_test, y_test = load_data()
    model_16, _, rec_16, kl_16 = exercise_1(x_train, x_val, x_test, y_test)
    exercise_2(model_16, rec_16, kl_16, x_train, x_val, x_test, y_test)


if __name__ == "__main__":
    main()
