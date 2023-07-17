import numpy as np
import matplotlib.pyplot as plt

def plot_2d_projections(matrix: np.ndarray, cmap=None) -> plt.Figure:
    """
    Plot the mean projections of a 3D matrix along the X, Y, and Z axes.

    Args:
        matrix (np.ndarray): A 3D matrix to plot.

    Returns:
        plt.Figure: A matplotlib Figure object containing the 2D projections along the X, Y, and Z axes.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(np.flip(matrix.mean(axis=0).T, 0), cmap=cmap)
    ax2.imshow(np.flip(matrix.mean(axis=1).T, 0), cmap=cmap)
    ax3.imshow(np.flip(matrix.mean(axis=2).T, 0), cmap=cmap)

    return fig