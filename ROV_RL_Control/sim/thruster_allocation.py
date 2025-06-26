import numpy as np
from typing import Optional

from sim.parameters import ROVParameters


class ThrusterAllocator:
    """
    ThrusterAllocator maps RL actions or desired 6-DOF force/moment vectors
    to individual thruster commands and back:
      1. map_action_to_thrust: [0,1]^8 -> actual thrusts [N]
      2. map_force_to_action: desired 6-vector -> normalized actions [0,1]^8
      3. allocate: direct action -> 6-DOF force/moment

    Attributes:
        P_max: maximum power per thruster [W]
        k, n: empirical coefficients for P->F: F = k * P^n
        max_thrusts: per-thruster thrust limits [N]
        T: thrust allocation matrix (6×8)
        T_pinv: pseudoinverse (8×6), computed if not provided
    """
    def __init__(self, params: Optional[ROVParameters] = None) -> None:
        # Load parameters
        self.params = params or ROVParameters()
        self.P_max: float = float(getattr(self.params, 'P_max', 240.0))
        self.k: float = float(getattr(self.params, 'k', 1.5801))
        self.n: float = float(getattr(self.params, 'n', 0.5819))
        # Avoid division by zero
        if self.n == 0:
            raise ValueError("Exponent n must be non-zero")
        self._inv_n: float = 1.0 / self.n

        # Thruster limits
        self.max_thrusts: np.ndarray = np.array(
            getattr(self.params, 'max_thrusts', [0.0]*8), dtype=np.float32)
        # Allocation matrices
        self.T: np.ndarray = np.array(
            getattr(self.params, 'thrust_allocation_matrix', np.zeros((6,8))), dtype=np.float32)
        if hasattr(self.params, 'thrust_allocation_pinv'):
            self.T_pinv: np.ndarray = np.array(
                self.params.thrust_allocation_pinv, dtype=np.float32)
        else:
            # Compute robust pseudoinverse
            self.T_pinv: np.ndarray = np.linalg.pinv(self.T)

    def map_action_to_thrust(self, action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action [0,1]^8 to actual thrust [N].
        """
        a = np.asarray(action, dtype=np.float32).flatten()
        if a.size != self.max_thrusts.size:
            raise ValueError(f"Action must be length {self.max_thrusts.size}, got {a.size}")
        # Clip normalized input
        a_clipped = np.clip(a, 0.0, 1.0)
        # Compute power and thrust
        P = a_clipped * self.P_max
        thrusts = self.k * np.power(P, self.n)
        # Enforce thrust limits
        return np.minimum(thrusts, self.max_thrusts)

    def map_force_to_action(self, force_moment: np.ndarray, clamp: bool = True) -> np.ndarray:
        """
        Invert desired 6-DOF force/moment -> normalized actions [0,1]^8.
        """
        f = np.asarray(force_moment, dtype=np.float32).reshape(6,)
        # Compute required thruster forces via pseudoinverse
        thrusts = self.T_pinv.dot(f)
        if clamp:
            thrusts = np.clip(thrusts, 0.0, self.max_thrusts)
        else:
            thrusts = np.clip(thrusts, 0.0, None)
        # Invert F = k * P^n  => P = (F/k)^(1/n)
        P = np.power(np.maximum(thrusts, 0.0) / self.k, self._inv_n)
        # Limit power and normalize
        P_clipped = np.clip(P, 0.0, self.P_max)
        return (P_clipped / self.P_max).astype(np.float32)

    def allocate(self, action: np.ndarray) -> np.ndarray:
        """
        Direct mapping: action -> thrusts -> force/moment (6-vector)
        """
        thrusts = self.map_action_to_thrust(action)
        return self.T.dot(thrusts)

    def inverse_allocation_error(self, force_moment: np.ndarray) -> float:
        """
        Compute reconstruction error || T * (T_pinv * f) - f ||_2.
        """
        f = np.asarray(force_moment, dtype=np.float32).reshape(6,)
        recon = self.T.dot(self.T_pinv.dot(f))
        return float(np.linalg.norm(recon - f))
