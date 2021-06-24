import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


def significant_digits(x, n=2):
    return np.format_float_positional(x, precision=n, fractional=False)


def plot_loss_curves(res: dict, ax: Axes):
    ax.plot(res['history'].history["loss"], linewidth=3, label='Training loss')
    ax.plot(res['history'].history["val_loss"], linewidth=3, label='Validation loss')
    plt.ylim([-0.01,0.31])
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    title = (f'Sequence Length {res["seq_length"]}\n'
             f'Test loss (MSE) = {significant_digits(res["test_loss"])}')
    plt.title(title)
    sns.despine(bottom=True)
    return


def plot_loss_vs_seq_length(results):
    plt.plot([res['test_loss'] for res in results],
             'x-',
             linewidth=3,
             label='Test loss',
             markersize=15,
             markeredgewidth=3,
             color='c')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('Sequence length')
    plt.title(f'Test loss (MSE)\ndepending on sequence length')
    sns.despine()
    plt.xticks(range(len(results)), [res['seq_length'] for res in results])
