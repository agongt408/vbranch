import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_line(results, branches, n_trials=8, p1=[], p2=None, show=True):
    shared_frac = [0, 0.25, 0.5, 0.75, 1]

    if len(p1) == 0:
        # baseline_acc = results['baseline'][0]
        # baseline_acc = results['baseline'][str(branches)][0]
        vbranch_acc = []
        vbranch_std = []

        for s in shared_frac:
            acc, std = results['vbranch'][str(branches)][str(s)]
            vbranch_acc.append(acc)
            vbranch_std.append(std)

        error = np.array(vbranch_std) / np.sqrt(n_trials)
        plt.errorbar(shared_frac, vbranch_acc, error*2, label='vbranch')
        # plt.plot(shared_frac, [baseline_acc]*len(shared_frac),
        #          label='baseline', linestyle='--')
    else:
        for p in p1:
            if p2 is None:
                baseline_acc = results['baseline'][str(p)][0]
            else:
                baseline_acc = results['baseline'][str(p)][str(p2)][0]
            vbranch_acc = []
            vbranch_std = []

            for s in shared_frac:
                if p2 is None:
                    acc, std = results['vbranch'][str(p)][str(branches)][str(s)]
                else:
                    acc, std = results['vbranch'][str(p)][str(p2)][str(branches)][str(s)]
                vbranch_acc.append(acc)
                vbranch_std.append(std)

            error = np.array(vbranch_std) / np.sqrt(n_trials)
            plt.errorbar(shared_frac, vbranch_acc, error*2, label=str(p))
            plt.plot(shared_frac, [baseline_acc]*len(shared_frac),
                     label=str(p), linestyle='--')

    plt.title(p2)
    plt.legend()
    if show:
        plt.show()

def plot_baseline_3d(results, X=[0.01, 0.05, 0.1, 0.2], Y=[8, 16, 32, 64, 128],
                     xlabel=None, ylabel=None):
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z = np.zeros((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            p1, p2 = X[i], Y[j]
            # Get mean value
            Z[j, i] = results[str(p1)][str(p2)][0]

    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Accuracy')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_ensemble_3d(results, X=[0.01, 0.05, 0.1, 0.2], p2=32, max_models=6, xlabel=None):
    # Make data
    Y = np.arange(2, max_models + 1) # Number of models in ensemble
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z = np.zeros((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            p1, branches = X[i], Y[j]
            Z[j, i] = results[str(p1)][str(p2)][str(branches)][0]

    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ensemble size')
    ax.set_zlabel('Accuracy')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_vbranch_3d(results, branches, X=[0.01, 0.05, 0.1, 0.2], p2=32, xlabel=None):
    # Make data
    Y = [0, 0.25, 0.5, 0.75, 1]
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z = np.zeros((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            p1, shared_frac = X[i], Y[j]
            Z[j, i] = results[str(p1)][str(p2)][str(branches)][str(shared_frac)][0]

    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.title('Branches: ' + str(branches))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Shared frac')
    ax.set_zlabel('Accuracy')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
