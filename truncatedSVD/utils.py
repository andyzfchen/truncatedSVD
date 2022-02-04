from numpy.linalg import svd
import matplotlib.pyplot as plt
from os import mkdir
from os.path import normpath, exists, join


def check_and_create_dir(dirname):
    """Check if directory exists. If it does not exist, create it.
    
    Parameters
    ----------
    dirname : str
        Name of directory
    """
    if not exists(normpath(dirname)):
        mkdir(normpath(dirname))

        
def get_truncated_svd(A, k):
    """Get truncated SVD using a deterministic method.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Real matrix
    
    k : int 
        Rank
    """
    u, s, vh = svd(A, full_matrices=False)
    return u[:, :k], s[:k], vh[:k, :]

    
def init_figure(title, xlabel, ylabel, yscale="log", figsize=(4, 3)):
    """Initialize matplotlib figure
    
    Parameters
    ----------
    title : str
        Figure title
    
    xlabel : str
        x-axis label
    
    ylabel : str
        y-axis label
    
    yscale : str, default="log"
        y-axis scale
    
    figsize : tuple, default=(4, 3)
        Figure size
    
    Returns
    -------
    fig : Figure
        matploltib Figure object
        
    ax : axes
        Matplotlib Axes object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Label figure
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    return fig, ax