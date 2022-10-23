"""Plot handling module.
"""
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def plot_snapshot(snapshot_state, r_in, r_out, progress_bar=False):
    fig, ax = plt.subplots()
    _plot_snapshot_core(snapshot_state, r_in, r_out, ax, progress_bar)
    fig.tight_layout()
    return fig


def plot_propagation_animation(state, r_in, r_out,
                               save_name=None,
                               progress_bar=False):
    """WON'T WORK
    """

    fig, ax = plt.subplots()

    def create_animation(i):
        _plot_snapshot_core(state[i], r_in, r_out, ax, progress_bar)

    ani = animation.FuncAnimation(fig, create_animation,
                                  interval=100)

    if save_name is None:
        plt.show()
    else:
        ani.save(save_name)


def _plot_snapshot_core(snapshot_state, r_in, r_out, ax,
                        progress_bar=False):
    num_anulus, num_segments = snapshot_state.shape
    size = (r_out - r_in) / num_anulus

    cmap = plt.colormaps["jet"]

    for i, anulus in enumerate(tqdm(snapshot_state,
                                    disable=not(progress_bar))):
        colors = cmap(anulus)
        radius = r_out - i * size
        ax.pie(anulus, radius=radius, colors=colors,
               wedgeprops=dict(width=size))
    ax.set_xlim(-r_out, r_out)
    ax.set_ylim(-r_out, r_out)

    return ax
