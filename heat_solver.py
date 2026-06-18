import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def apply_bc(T):
    # 1. Handle Neumann (insulated sides) first. This lets corners copy inner values...
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]

    # 2. ...and then Dirichlet overrides the corners so they stay strictly 0.0
    T[0, :] = 0.0
    T[-1, :] = 0.0


def solve_heat_2d(nx=50, ny=50, Lx=1.0, Ly=1.0, alpha=1e-4, dt=0.1, nt=500):
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Quick stability check
    dt_max = 1 / (2 * alpha * (1 / dx**2 + 1 / dy**2))
    if dt > dt_max:
        raise ValueError(f"dt too large! Max stable dt = {dt_max:.4e}")

    # Set up initial grid (rows = y, cols = x)
    T = np.zeros((ny, nx))
    T[ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4] = 100.0

    history = [T.copy()]

    for _ in range(nt):
        Tn = T.copy()

        # Center differences for the inner grid
        T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + alpha * dt * (
            (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / dx**2
            + (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / dy**2
        )

        apply_bc(T)
        history.append(T.copy())

    return history


def animate_solution(history):
    fig, ax = plt.subplots()

    # Fixed vmin/vmax stops matplotlib from auto-adjusting the color scale
    cax = ax.imshow(history[0], cmap="hot", origin="lower", vmin=0, vmax=100)
    fig.colorbar(cax, label="Temperature")

    def update(frame):
        cax.set_array(history[frame])
        ax.set_title(f"Time Step: {frame}")
        return (cax,)

    anim = FuncAnimation(
        fig, update, frames=len(history), interval=20, blit=True
    )
    plt.show()


if __name__ == "__main__":
    history = solve_heat_2d()
    animate_solution(history)
plt.imshow(history[100], cmap="hot", origin="lower", vmin=0, vmax=100)
plt.colorbar()
plt.show()
print(history[0].max(), history[-1].max())
