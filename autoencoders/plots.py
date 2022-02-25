import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from autoencoders.utils import choose_z


def plot_grouped_history(history):
    if isinstance(history, dict):
        history = pd.DataFrame(history)
    val_metrics = list()
    for column in history.columns:
        if "val_" in column:
            val_metrics.append(column.replace("val_", ""))
            val_metrics.append(column)
    for i in range(0, len(val_metrics), 2):
        plt.title(val_metrics[i])
        plt.plot(history[val_metrics[i]], label="train")
        plt.plot(history[val_metrics[i + 1]], label="val")
        plt.legend()
        plt.show()
    for column in history.columns:
        if column not in val_metrics:
            plt.title(column)
            plt.plot(history[column], label="train")
            plt.legend()
            plt.show()


def plot_latent_space(
        Z,
        y=None,
        title="",
        labels_names=None,
        figsize=(5, 5),
        dpi=96,
        legend=True,
        fontsize=20,
        labelsize=20,
        legendfontsize=15,
):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b',
              u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    plt.figure(figsize=figsize, dpi=dpi)
    #plt.suptitle(title)
    plt.title(title, fontsize=fontsize)
    if Z.shape[1] != 2:
        warnings.warn("Latent space is not bidimentional, only the first 2 dimensions will be plotted.")
    if y is None:
        plt.scatter(Z[:, 0], Z[:, 1])
    else:
        for label in np.unique(y):
            idxs = np.nonzero(y == label)
            plt.scatter(
                Z[:, 0][idxs],
                Z[:, 1][idxs],
                c=colors[label % len(colors)],
                label=label if labels_names is None else labels_names[label]
            )
    #plt.xlim([-3, 3])
    #plt.ylim([-3, 3])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.gca().set_aspect('equal', adjustable='box')
    if legend:
        plt.legend(loc="lower right", fontsize=legendfontsize, framealpha=0.95)
    """if labels is not None:
        for i, label in enumerate(labels):
            legend.get_texts()[i].set_text(label)"""
    plt.show()


def plot_reconstruction(X, encoder, decoder, figsize=(20, 15), n=0):
    X_tilde = decoder.predict(encoder.predict(X))
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_reconstruction_vae(X, encoder, decoder, figsize=(20, 15), n=0):
    Z = list()
    for x in X:
        z = choose_z(x[np.newaxis, :, :], encoder, decoder)
        Z.append(z.ravel())
    Z = np.array(Z)
    X_tilde = decoder.predict(Z)
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_labeled_latent_space_matrix(
        Z,
        y,
        **kwargs
):
    Z = list(Z)
    Z = pd.DataFrame(Z)
    pd.plotting.scatter_matrix(Z,
                               c=y,
                               cmap="viridis",
                               diagonal="kde",
                               alpha=1,
                               s=100,
                               figsize=kwargs.get("figsize", (8, 8)))