# visualization/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def plot_trajectory(
    positions: np.ndarray,
    targets: np.ndarray,
    save_path: str,
    title: str = "Trajectory Comparison"
) -> None:
    """
    Plot actual trajectory vs target positions in 3D (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Actual')
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], '--', label='Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error(
    errors: np.ndarray,
    save_path: str,
    title: str = "Tracking Error"
) -> None:
    """
    Plot tracking error over time.
    """
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Timestep')
    plt.ylabel('Position Error (m)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_control_inputs(
    inputs: np.ndarray,
    save_path: str,
    title: str = "Control Inputs"
) -> None:
    """
    Plot control actions (normalized thruster commands) over time.
    """
    T, N = inputs.shape
    plt.figure()
    for i in range(N):
        plt.plot(np.arange(T), inputs[:, i], label=f'Input {i}')
    plt.xlabel('Timestep')
    plt.ylabel('Action Value')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_episode(
    episode_data: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Generate and save trajectory, error, and control input plots for a single episode.

    episode_data keys:
      - 'positions': list of (3,) arrays
      - 'targets': list of (3,) arrays
      - 'errors': list of floats
      - 'actions': list of (N,) arrays
    """
    # Convert lists to numpy arrays for plotting
    positions = np.asarray(episode_data['positions'], dtype=np.float32)
    targets = np.asarray(episode_data['targets'], dtype=np.float32)
    errors = np.asarray(episode_data['errors'], dtype=np.float32)
    actions = np.asarray(episode_data['actions'], dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    plot_trajectory(
        positions, targets,
        os.path.join(output_dir, 'trajectory.png')
    )
    plot_error(
        errors,
        os.path.join(output_dir, 'error.png')
    )
    plot_control_inputs(
        actions,
        os.path.join(output_dir, 'actions.png')
    )
