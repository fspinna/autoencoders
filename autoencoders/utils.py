import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import pathlib


def make_path(folder):
    path = pathlib.Path(folder)
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)
    return path


def get_project_root():
    return pathlib.Path(__file__).parent


def choose_z(x, encoder, decoder, n=1000, x_label=None, blackbox=None, check_label=False, verbose=False, mse=False):
    X = np.repeat(x, n, axis=0)
    Z = encoder.predict(X)
    Z_tilde = decoder.predict(Z)
    if check_label:
        y_tilde = blackbox.predict(Z_tilde)
        y_correct = np.nonzero(y_tilde == x_label)
        if len(Z_tilde[y_correct]) == 0:
            if verbose:
                warnings.warn("No instances with the same label of x found.")
        else:
            Z_tilde = Z_tilde[y_correct]
            Z = Z[y_correct]
    if mse:
        distances = []
        for z_tilde in Z_tilde:
            distances.append(((x - z_tilde) ** 2).sum())
        distances = np.array(distances)
    else:
        # distances = cdist(x[:, :, 0], Z_tilde[:, :, 0]) # does not work for multi ts
        distances = cdist(x.reshape(-1, x.shape[1] * x.shape[2]), Z_tilde.reshape(-1, Z_tilde.shape[1] *
                                                                                  Z_tilde.shape[2]))
    best_z = Z[np.argmin(distances)]
    return best_z.reshape(1, -1)


def reconstruction_accuracy(X, encoder, decoder, blackbox, repeat=1, verbose=True):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        y_tilde = blackbox.predict(decoder.predict(encoder.predict(X)))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def reconstruction_accuracy_vae(X, encoder, decoder, blackbox, repeat=1, n=100, check_label=True, verbose=True):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        Z = list()
        for x in X:
            if check_label:
                x_label = blackbox.predict(x[np.newaxis, :, :])
            else:
                x_label = None
            z = choose_z(x=x[np.newaxis, :, :],
                         encoder=encoder,
                         decoder=decoder,
                         n=n,
                         x_label=x_label,
                         blackbox=blackbox,
                         check_label=check_label,
                         verbose=verbose)
            Z.append(z.ravel())
        Z = np.array(Z)
        y_tilde = blackbox.predict(decoder.predict(Z))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


