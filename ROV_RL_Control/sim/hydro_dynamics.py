import numpy as np
from sim.parameters import ROVParameters


class HydroDynamics:
    """
    6-DOF rigid-body hydrodynamics with quaternion representation.
    Dynamics: M ν̇ + C(ν) ν + D(ν) ν + g(η) = τ
    State vector: [pos(3), quat(4), v(3), ω(3)]
    """
    def __init__(self, params: ROVParameters = None):
        self.params = params or ROVParameters()
        # Inverse mass-inertia matrix (6×6)
        self.M_inv: np.ndarray = self.params.M_inv.astype(np.float64)
        # Optional Coriolis and damping callbacks
        self.C_func = getattr(self.params, 'C_func', None)
        self.D_func = getattr(self.params, 'D_func', None)
        # Precompute gravity and buoyancy effect
        self.g_vec: np.ndarray = self._compute_gravity_buoyancy()
        # Time step
        self.dt: float = float(self.params.dt)

    def _compute_gravity_buoyancy(self) -> np.ndarray:
        """
        Compute gravity minus buoyancy generalized force vector.
        Returns a 6D vector: [0,0, m*g - B, 0,0,0].
        """
        # Mass fallback
        m = getattr(self.params, 'mass', None)
        if m is None:
            m = getattr(self.params, 'm', None)
        if m is None:
            raise AttributeError("ROVParameters missing 'mass' or 'm'")
        # Gravity constant
        gravity = getattr(self.params, 'g', 9.81)
        # Buoyancy fallback
        B = getattr(self.params, 'buoyancy', None)
        if B is None:
            B = getattr(self.params, 'B', 0.0)
        # Build vector
        g_vec = np.zeros(6, dtype=np.float64)
        g_vec[2] = m * gravity - B
        return g_vec

    @staticmethod
    def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
        """Convert unit quaternion to rotation matrix."""
        qw, qx, qy, qz = (q / np.linalg.norm(q)).astype(np.float64)
        return np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx**2+qy**2)]
        ], dtype=np.float32)

    @staticmethod
    def omega_to_quat_mat(omega: np.ndarray) -> np.ndarray:
        """Return Ω(ω) matrix for quaternion kinematics."""
        wx, wy, wz = omega.astype(np.float64)
        return np.array([
            [ 0. , -wx, -wy, -wz],
            [ wx,  0. ,  wz, -wy],
            [ wy, -wz,  0. ,  wx],
            [ wz,  wy, -wx,  0. ]
        ], dtype=np.float64)

    def acceleration(self, nu: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute generalized acceleration: ν̇ = M_inv (τ - g - Cν - Dν).
        nu: [u,v,w,p,q,r], tau: [Fx,Fy,Fz,Mx,My,Mz]
        """
        rhs = tau.astype(np.float64) - self.g_vec
        if self.C_func is not None:
            rhs -= self.C_func(nu).dot(nu)
        if self.D_func is not None:
            rhs -= self.D_func(nu).dot(nu)
        return self.M_inv.dot(rhs)

    def deriv(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Compute state derivative: [pos_dot, quat_dot, v_dot, omega_dot].
        """
        pos = state[0:3].astype(np.float64)
        quat = state[3:7].astype(np.float64)
        v_lin = state[7:10].astype(np.float64)
        omega = state[10:13].astype(np.float64)

        # Kinematics
        R = self.quaternion_to_rotation(quat)
        pos_dot = R.dot(v_lin)
        quat_dot = 0.5 * self.omega_to_quat_mat(omega).dot(quat)

        # Dynamics
        nu = np.concatenate([v_lin, omega])
        nu_dot = self.acceleration(nu, tau)
        v_dot = nu_dot[0:3]
        omega_dot = nu_dot[3:6]

        return np.concatenate([pos_dot, quat_dot, v_dot, omega_dot]).astype(np.float32)

    def integrate(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Integrate state forward dt using RK4 and normalize quaternion.
        """
        h = self.dt
        k1 = self.deriv(state, tau)
        k2 = self.deriv(state + 0.5*h*k1, tau)
        k3 = self.deriv(state + 0.5*h*k2, tau)
        k4 = self.deriv(state +   h*k3, tau)
        next_state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # Normalize quaternion
        q = next_state[3:7]
        next_state[3:7] = (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)
        return next_state
