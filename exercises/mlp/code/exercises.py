"""Runs Exercises 2, 3 and 4: data, training, Figures 1-6 and every reported number.

Usage: python exercises.py

Exercises 2, 3 and 4A all draw from the single rng = default_rng(42). Exercise 4B
needs several seeds, so seed 42 is those same runs and the other seeds get their
own generators.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP, SigmoidBCE, SoftmaxCCE, one_hot

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_TINT, ORANGE_TINT = "#cde2fb", "#fbdccf"
INK, SECONDARY_INK = "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f5f9fe", "#9ec5f4", "#3987e5", "#104281"]
)

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

# Exercise 2
EX2_SIZES, EX2_LR, EX2_EPOCHS = [2, 4, 1], 0.1, 1000
# Exercises 3 and 4: depth is the only thing that changes
EX3_SIZES, EX4_SIZES = [4, 16, 3], [4, 16, 16, 3]
EX3_LR, EX3_EPOCHS = 0.1, 2000
EXTRA_SEEDS = [43, 44, 45, 46]


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)


def scatter_classes(ax, X, y, labels, size=20):
    """Scatter both classes with a thin surface ring so overlapping points stay legible."""
    for cls, color in enumerate((BLUE, ORANGE)):
        ax.scatter(
            X[y == cls, 0],
            X[y == cls, 1],
            s=size,
            c=color,
            edgecolors="white",
            linewidths=0.4,
            label=labels[cls],
        )


def plot_loss(history, title, ylabel, label, name):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history, color=BLUE, label=label)
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.legend()
    save(fig, name)


def exercise_2(rng):
    print("=== Exercise 2 ===")
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(
        n_samples=[500, 250, 250], centers=centers, cluster_std=1.2, random_state=42
    )
    y = (c > 0).astype(int)  # class 0: 1 cluster | class 1: 2 clusters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Figure 1: the full dataset
    fig, ax = plt.subplots(figsize=(5.2, 5))
    scatter_classes(ax, X, y, ["Class 0 (1 cluster)", "Class 1 (2 clusters)"])
    ax.set(
        title="Figure 1: Class 1 sits on both sides of Class 0",
        xlabel="Feature $x_1$",
        ylabel="Feature $x_2$",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(fig, "fig1.png")

    # Item B: linear baseline
    baseline = LogisticRegression().fit(X_train, y_train)
    print(f"Baseline test accuracy: {baseline.score(X_test, y_test):.4f}")
    print(f"Baseline coefficients: {baseline.coef_.round(4)}, "
          f"intercept: {baseline.intercept_.round(4)}")
    print(f"Train mean of class 0: {X_train[y_train == 0].mean(axis=0).round(4)}")
    print(f"Train mean of class 1: {X_train[y_train == 1].mean(axis=0).round(4)}")

    # Item C: the MLP
    net = MLP(EX2_SIZES, SigmoidBCE(), rng)
    history = net.fit(X_train, y_train[:, None], EX2_LR, EX2_EPOCHS)
    print(f"Architecture {EX2_SIZES}, lr {EX2_LR}, {EX2_EPOCHS} epochs, "
          f"{net.n_params()} parameters")
    print(f"Final train loss: {history[-1]:.4f}")
    print(f"Train accuracy: {net.accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy: {net.accuracy(X_test, y_test):.4f}")

    plot_loss(
        history,
        "Figure 2: Exercise 2 training loss",
        "Binary cross-entropy",
        f"MLP {EX2_SIZES}",
        "fig2.png",
    )

    # Where the boundary crosses the diagonal x1 = x2, the line through both Class 1 clusters
    t = np.linspace(-8, 8, 16001)
    u_diagonal = net.forward(np.c_[t, t])[0][:, 0]
    crossings = t[:-1][np.diff(np.sign(u_diagonal)) != 0]
    print(f"Boundary crosses x1 = x2 at: {crossings.round(2)}")

    # Figure 3: decision regions over the test points. u = 0 is the p = 0.5 boundary.
    x1, x2 = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(-8, 8, 400))
    u = net.forward(np.c_[x1.ravel(), x2.ravel()])[0].reshape(x1.shape)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.contourf(x1, x2, u, levels=[u.min(), 0, u.max()], colors=[BLUE_TINT, ORANGE_TINT])
    ax.contour(x1, x2, u, levels=[0], colors=SECONDARY_INK, linewidths=0.8)
    scatter_classes(ax, X_test, y_test, ["Test point, Class 0", "Test point, Class 1"], size=24)
    ax.set(
        title="Figure 3: decision regions of the MLP",
        xlabel="Feature $x_1$",
        ylabel="Feature $x_2$",
    )
    ax.set_aspect("equal")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(color=BLUE_TINT, label="Region predicted Class 0"),
        Patch(color=ORANGE_TINT, label="Region predicted Class 1"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig3.png")


def classification_data(shuffle=True):
    return make_classification(
        n_samples=1500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=1.0,
        shuffle=shuffle,
        random_state=42,
    )


def run_network(sizes, data, rng, epochs=EX3_EPOCHS):
    """Train one network with the shared Exercise 3 budget and collect its results."""
    X_train, X_test, y_train, y_test = data
    net = MLP(sizes, SoftmaxCCE(), rng)
    history = net.fit(X_train, one_hot(y_train, 3), EX3_LR, epochs)
    return {
        "net": net,
        "history": history,
        "loss": history[-1],
        "train_acc": net.accuracy(X_train, y_train),
        "test_acc": net.accuracy(X_test, y_test),
    }


def most_confused_pair(matrix):
    """Return the class pair with the most errors, counting both directions."""
    errors = matrix + matrix.T
    np.fill_diagonal(errors, 0)
    a, b = np.unravel_index(errors.argmax(), errors.shape)
    return min(a, b), max(a, b), errors


def cluster_structure(X, X_test, predictions):
    """Trace the Exercise 3 errors back to the six true clusters.

    With shuffle=False the generator returns the same points in generation order:
    250 per cluster, where cluster k belongs to class k % 3. Shuffling only permutes
    rows and features, and neither changes the distances between points.
    """
    X_ordered, y_ordered = classification_data(shuffle=False)
    same_points = np.allclose(
        np.sort(np.sort(X_ordered, axis=1), axis=0), np.sort(np.sort(X, axis=1), axis=0)
    )
    print(f"shuffle=False gives the same points: {same_points}")

    size = 250
    clusters = [X_ordered[k * size : (k + 1) * size] for k in range(6)]
    majority = [int(np.bincount(y_ordered[k * size : (k + 1) * size]).argmax()) for k in range(6)]
    print(f"Majority class of clusters 0-5: {majority}")

    centers = np.array([c.mean(axis=0) for c in clusters])
    spread = [np.sqrt(((c - c.mean(axis=0)) ** 2).sum(axis=1).mean()) for c in clusters]
    print(f"Cluster spread (rms distance to own center): {np.round(spread, 2)}")
    print("Center distances between clusters 0-5:")
    print(np.linalg.norm(centers[:, None] - centers[None, :], axis=2).round(2))

    # Overlap: how often a point's nearest neighbour lives in another cluster
    distances = np.linalg.norm(X_ordered[:, None] - X_ordered[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbour_cluster = distances.argmin(axis=1) // size
    links = np.zeros((6, 6), dtype=int)
    for i, j in zip(np.arange(1500) // size, neighbour_cluster):
        links[i, j] += 1
    print("Nearest-neighbour links between clusters 0-5:")
    print(links)

    # Which cluster did each test point come from, and what did the network say?
    key = lambda row: tuple(np.round(np.sort(row), 9))
    cluster_of = {key(row): i // size for i, row in enumerate(X_ordered)}
    test_cluster = np.array([cluster_of[key(row)] for row in X_test])
    print("Test predictions by true cluster (columns: predicted class 0, 1, 2):")
    for k in range(6):
        counts = np.bincount(predictions[test_cluster == k], minlength=3)
        print(f"  cluster {k} (class {k % 3}): {counts}")


def exercise_3(rng):
    print("\n=== Exercise 3 ===")
    X, y = classification_data()
    data = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = data

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(f"Baseline test accuracy: {baseline.score(X_test, y_test):.4f}")

    run = run_network(EX3_SIZES, data, rng)
    print(f"Architecture {EX3_SIZES}, lr {EX3_LR}, {EX3_EPOCHS} epochs, "
          f"{run['net'].n_params()} parameters")
    print(f"Final train loss: {run['loss']:.4f}")
    print(f"Train accuracy: {run['train_acc']:.4f}")
    print(f"Test accuracy: {run['test_acc']:.4f}")

    plot_loss(
        run["history"],
        "Figure 4: Exercise 3 training loss",
        "Categorical cross-entropy",
        f"MLP {EX3_SIZES}",
        "fig4.png",
    )

    # Figure 5: confusion matrix on the test set
    predictions = run["net"].predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix (rows: true, columns: predicted):")
    print(matrix)
    a, b, errors = most_confused_pair(matrix)
    print(f"Most-confused pair: classes {a} and {b}, {errors[a, b]} errors")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"Errors between classes {i} and {j}: {errors[i, j]}")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matrix, cmap=BLUES)
    names = [f"Class {i}" for i in range(3)]
    ax.set(
        title="Figure 5: Exercise 3 confusion matrix (test set)",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(3),
        yticks=range(3),
        xticklabels=names,
        yticklabels=names,
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(3):
        for j in range(3):
            dark_cell = matrix[i, j] > matrix.max() * 0.55
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if dark_cell else INK)
    save(fig, "fig5.png")

    cluster_structure(X, X_test, predictions)
    return data, run


def exercise_4(rng, data, shallow_run):
    print("\n=== Exercise 4 ===")
    deep_run = run_network(EX4_SIZES, data, rng)
    print(f"Architecture {EX4_SIZES}, lr {EX3_LR}, {EX3_EPOCHS} epochs, "
          f"{deep_run['net'].n_params()} parameters")
    print(f"Final train loss: {deep_run['loss']:.4f}")
    print(f"Train accuracy: {deep_run['train_acc']:.4f}")
    print(f"Test accuracy: {deep_run['test_acc']:.4f}")

    # Figure 6: both curves on the same axes
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for run, color, label in (
        (shallow_run, BLUE, f"Exercise 3: {EX3_SIZES}"),
        (deep_run, ORANGE, f"Exercise 4: {EX4_SIZES}"),
    ):
        ax.plot(run["history"], color=color, label=label)
        ax.annotate(
            f"{run['loss']:.3f}",
            xy=(EX3_EPOCHS, run["loss"]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    ax.set(
        title="Figure 6: one hidden layer against two (seed 42)",
        xlabel="Epoch",
        ylabel="Categorical cross-entropy",
    )
    ax.legend()
    save(fig, "fig6.png")

    # Item B: same comparison over several seeds. Seed 42 is the runs above.
    runs = {"Exercise 3": [shallow_run], "Exercise 4": [deep_run]}
    for seed in EXTRA_SEEDS:
        seed_rng = np.random.default_rng(seed)
        runs["Exercise 3"].append(run_network(EX3_SIZES, data, seed_rng))
        runs["Exercise 4"].append(run_network(EX4_SIZES, data, seed_rng))

    seeds = [42] + EXTRA_SEEDS
    print(f"Seeds: {seeds}")
    for name, results in runs.items():
        test = np.array([r["test_acc"] for r in results])
        train = np.array([r["train_acc"] for r in results])
        loss = np.array([r["loss"] for r in results])
        print(f"{name}: {results[0]['net'].n_params()} parameters")
        print(f"  test accuracy per seed: {test.round(4)}")
        print(f"  final train loss per seed: {loss.round(4)}")
        print(f"  mean final train loss {loss.mean():.4f} (std {loss.std():.4f})")
        print(f"  mean train accuracy {train.mean():.4f} (std {train.std():.4f})")
        print(f"  mean test accuracy {test.mean():.4f} (std {test.std():.4f})")

    shallow_test = np.array([r["test_acc"] for r in runs["Exercise 3"]])
    deep_test = np.array([r["test_acc"] for r in runs["Exercise 4"]])
    differences = deep_test - shallow_test
    print(f"Test accuracy difference (Ex. 4 - Ex. 3) per seed: {differences.round(4)}")
    print(f"Mean difference: {differences.mean():+.4f} (std {differences.std():.4f})")

    X_test, y_test = data[1], data[3]
    pairs = [
        most_confused_pair(confusion_matrix(y_test, r["net"].predict(X_test)))[:2]
        for r in runs["Exercise 3"]
    ]
    print(f"Most-confused pair per seed, Exercise 3 network: {[tuple(map(int, p)) for p in pairs]}")

    # Controls outside the required comparison: the [4, 16, 16, 3] network also has
    # three times the parameters, so match the parameter counts to isolate depth,
    # then triple the budget to see whether train loss and test accuracy agree.
    print("Controls (fresh generator per seed, same seeds, same learning rate):")
    for sizes, epochs in (
        ([4, 16, 3], EX3_EPOCHS),
        ([4, 8, 8, 3], EX3_EPOCHS),
        ([4, 50, 3], EX3_EPOCHS),
        ([4, 16, 16, 3], EX3_EPOCHS),
        ([4, 16, 3], 3 * EX3_EPOCHS),
        ([4, 16, 16, 3], 3 * EX3_EPOCHS),
    ):
        results = [run_network(sizes, data, np.random.default_rng(s), epochs) for s in seeds]
        print(
            f"  {str(sizes):15s} {epochs:5d} epochs, "
            f"{results[0]['net'].n_params():3d} parameters: "
            f"train loss {np.mean([r['loss'] for r in results]):.4f}, "
            f"train acc {np.mean([r['train_acc'] for r in results]):.4f}, "
            f"test acc {np.mean([r['test_acc'] for r in results]):.4f}"
        )


def main():
    rng = np.random.default_rng(42)
    exercise_2(rng)
    data, shallow_run = exercise_3(rng)
    exercise_4(rng, data, shallow_run)


if __name__ == "__main__":
    main()
