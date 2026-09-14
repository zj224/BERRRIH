import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def fish_animation(p, Xinertial, save_path=None, animation=True):
    """Port of fishAnimation.m

    Animates the fish body (x,y node positions) over time and the path
    traced by the head node. If save_path is given, saves the animation
    to that file (e.g. 'fish.mp4' or 'fish.gif') instead of only
    displaying it interactively.

    If animation=False, plots only the final frame and the full head
    trajectory without running an animation.
    """
    x = Xinertial[p.ix, :]  # select x states for plotting
    y = Xinertial[p.iy, :]  # select y states for plotting

    x_min = np.min(x) - 0.1 * abs(np.min(x))
    x_max = np.max(x) + 0.1 * abs(np.max(x))
    y_min = np.min(y) - 0.1 * abs(np.min(y))
    y_max = np.max(y) + 0.1 * abs(np.max(y))

    fig, ax = plt.subplots(num=1)
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")

    (fig_animate,) = ax.plot(x[:, 0], y[:, 0], "b-", linewidth=5)  # fish body
    (h1,) = ax.plot(x[0, 0], y[0, 0], "b-")  # head trace

    if animation:
        def update(k):
            nonlocal h1
            fig_animate.set_data(x[:, k], y[:, k])
            h1.remove()
            j = np.arange(0, k)
            (h1,) = ax.plot(x[0, j], y[0, j], "b-")
            return fig_animate, h1

        anim = FuncAnimation(
            fig, update, frames=range(1, p.nt),
            interval=max(p.t_pause * 1000, 1), blit=False, repeat=False,
        )

        if save_path is not None:
            anim.save(save_path)
        else:
            plt.show()

        return anim

    # Static final-frame plot: draw the full head trajectory and the fish body
    # at the last timestep.
    last_idx = p.nt - 1
    fig_animate.set_data(x[:, last_idx], y[:, last_idx])
    h1.remove()
    j = np.arange(0, p.nt)
    (h1,) = ax.plot(x[0, j], y[0, j], "b-")

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()

    return None
