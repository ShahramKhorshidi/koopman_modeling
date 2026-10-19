import osqp
import numpy as np
from scipy import sparse


class MovingHorizonEstimator():
    def __init__(self, A_matrix, B_matric, C_matrix, N, state_bounds=None):
        """
        Initialize the Moving Horizon Estimator.

        Args:
            A_matrix (numpy.ndarray): System dynamics matrix.
            B_matrix (numpy.ndarray): Input matrix.
            C_matrix (numpy.ndarray): Measurement matrix.
            N (int): Estimation horizon length.
            state_bounds (tuple): Lower and upper bounds on the actual states.
        """
        A_matrix = np.asarray(A_matrix, dtype=float)
        B_matric = np.asarray(B_matric, dtype=float)
        C_matrix = np.asarray(C_matrix, dtype=float)
        self._check_system_matrices(A_matrix, B_matric, C_matrix)
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("The estimation horizon N must be a positive integer.")

        # System dimensions
        self.n_x = A_matrix.shape[0] # Number of state observables
        self.n_u = B_matric.shape[1] # Number of inputs
        self.n_y = C_matrix.shape[0] # Number of measurements
        
        # System Matrices        
        self.A = A_matrix
        self.B = B_matric
        self.C = C_matrix
        
        self.N = int(N)
        self.Sigma_0 = np.eye(self.n_x) # Arrival-cost weight
        self.Q_w = np.eye(self.n_x)     # Process noise weight
        self.R_v = np.eye(self.n_y)     # Measurement noise weight
        self.state_bounds = self._check_state_bounds(state_bounds)
        
        # Initialize input and measurement history
        self.U = np.zeros((self.n_u, self.N))
        self.Y = np.zeros((self.n_y, self.N))
        self.n_measurements = 0

        # The sparse QP structure is built once and only its linear cost is
        # updated when a new input/measurement pair arrives.
        self.problem = None
        self.problem_needs_rebuild = True
        self.solution = None
        self.warm_start_solution = np.zeros((self.n_x, self.N + 1))
        self.linear_cost = np.zeros((self.N + 1, self.n_x))

    @staticmethod
    def _check_system_matrices(A, B, C):
        if not all(np.all(np.isfinite(matrix)) for matrix in (A, B, C)):
            raise ValueError("System matrices must contain finite values.")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A_matrix must be a square matrix.")
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("B_matrix has an incompatible shape.")
        if C.ndim != 2 or C.shape[1] != A.shape[0]:
            raise ValueError("C_matrix has an incompatible shape.")

    def _check_state_bounds(self, state_bounds):
        if state_bounds is None:
            return None
        if len(state_bounds) != 2:
            raise ValueError("state_bounds must contain lower and upper bounds.")
        lower = np.asarray(state_bounds[0], dtype=float).reshape(-1)
        upper = np.asarray(state_bounds[1], dtype=float).reshape(-1)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("The state bounds must contain finite values.")
        if lower.size != upper.size or lower.size > self.n_x:
            raise ValueError("The state bounds have incompatible dimensions.")
        if np.any(lower > upper):
            raise ValueError("Every lower state bound must be below its upper bound.")
        return lower, upper

    @staticmethod
    def _check_weight(weight, dimension, name):
        weight = np.asarray(weight, dtype=float)
        if not np.all(np.isfinite(weight)):
            raise ValueError(f"{name} must contain finite values.")
        if weight.shape != (dimension, dimension):
            raise ValueError(f"{name} has an incompatible shape.")
        if not np.allclose(weight, weight.T):
            raise ValueError(f"{name} must be symmetric.")
        weight = 0.5 * (weight + weight.T)
        eigenvalues, eigenvectors = np.linalg.eigh(weight)
        if np.min(eigenvalues) < -1e-12:
            raise ValueError(f"{name} must be positive semi-definite.")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def set_arrival_cost_weight(self, Sigma_0):
        self.Sigma_0 = self._check_weight(
            Sigma_0, self.n_x, "Arrival-cost weight"
        )
        self.problem_needs_rebuild = True
    
    def set_process_noise_weight(self, Q_w):
        self.Q_w = self._check_weight(Q_w, self.n_x, "Process-noise weight")
        self.problem_needs_rebuild = True
        
    def set_measurement_noise_weight(self, R_v):
        self.R_v = self._check_weight(
            R_v, self.n_y, "Measurement-noise weight"
        )
        self.problem_needs_rebuild = True

    def get_horizon(self):
        return self.N
            
    def update_measurements(self, u_new, y_new):
        """
        Update the measurement and input history with new data.

        Args:
            u_new (numpy.ndarray): New control input.
            y_new (numpy.ndarray): New measurement.
        """
        u_new = np.asarray(u_new, dtype=float).reshape(-1)
        y_new = np.asarray(y_new, dtype=float).reshape(-1)
        if u_new.size != self.n_u:
            raise ValueError("The control input has an incompatible shape.")
        if y_new.size != self.n_y:
            raise ValueError("The measurement has an incompatible shape.")
        if not np.all(np.isfinite(u_new)) or not np.all(np.isfinite(y_new)):
            raise ValueError("Inputs and measurements must contain finite values.")

        # Each stored pair is (u_k, y_{k+1}).
        self.U[:, :-1] = self.U[:, 1:]
        self.U[:, -1] = u_new
        self.Y[:, :-1] = self.Y[:, 1:]
        self.Y[:, -1] = y_new
        self.n_measurements = min(self.n_measurements + 1, self.N)

    def _build_optimization_problem(self):
        """Build the constant block-tridiagonal part of the MHE QP."""
        n_variables = (self.N + 1) * self.n_x
        hessian = sparse.lil_matrix((n_variables, n_variables))

        self.A_transpose_Q = self.A.T @ self.Q_w
        self.C_transpose_R = self.C.T @ self.R_v
        A_transpose_Q_A = self.A_transpose_Q @ self.A
        C_transpose_R_C = self.C_transpose_R @ self.C

        # Scale the optimization variables because the position and momentum
        # precision weights can differ by many orders of magnitude.
        typical_diagonal = np.diag(
            self.Sigma_0
            + A_transpose_Q_A
            + self.Q_w
            + C_transpose_R_C
        )
        self.state_scale = np.ones(self.n_x)
        scaled_states = typical_diagonal > np.finfo(float).eps
        self.state_scale[scaled_states] = (
            1.0 / np.sqrt(typical_diagonal[scaled_states])
        )
        scaling_matrix = np.diag(self.state_scale)

        for k in range(self.N + 1):
            state_slice = slice(k * self.n_x, (k + 1) * self.n_x)
            diagonal_block = np.zeros((self.n_x, self.n_x))

            if k == 0:
                diagonal_block += self.Sigma_0
            if k < self.N:
                diagonal_block += A_transpose_Q_A
            if k > 0:
                diagonal_block += self.Q_w + C_transpose_R_C

            hessian[state_slice, state_slice] = (
                scaling_matrix @ diagonal_block @ scaling_matrix
            )

            if k < self.N:
                next_state_slice = slice(
                    (k + 1) * self.n_x,
                    (k + 2) * self.n_x,
                )
                cross_block = (
                    -scaling_matrix
                    @ self.A_transpose_Q
                    @ scaling_matrix
                )
                hessian[state_slice, next_state_slice] = cross_block
                hessian[next_state_slice, state_slice] = cross_block.T

        # OSQP uses 0.5 z.T P z + q.T z and only needs the upper triangle of P.
        P_matrix = sparse.triu(2.0 * hessian.tocsc(), format="csc")
        bound_matrix = sparse.eye(n_variables, format="csc")

        lower_bound = np.full(self.n_x, -np.inf)
        upper_bound = np.full(self.n_x, np.inf)
        if self.state_bounds is not None:
            lower, upper = self.state_bounds
            lower_bound[:lower.size] = lower
            upper_bound[:upper.size] = upper

        lower_bound = np.tile(
            lower_bound / self.state_scale, self.N + 1
        )
        upper_bound = np.tile(
            upper_bound / self.state_scale, self.N + 1
        )

        self.problem = osqp.OSQP()
        self.problem.setup(
            P=P_matrix,
            q=np.zeros(n_variables),
            A=bound_matrix,
            l=lower_bound,
            u=upper_bound,
            verbose=False,
            warm_start=True,
            polish=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            check_termination=10,
            max_iter=10000,
        )
        self.problem_needs_rebuild = False
        
    def setup_optimization_problem(self, x_0_hat):
        """
        Setup the optimization problem for the estimator.

        Args:
            x_0_hat (numpy.ndarray): Initial state estimate.
        """
        if self.n_measurements < self.N:
            raise RuntimeError("The input and measurement history is not full yet.")
        x_0_hat = np.asarray(x_0_hat, dtype=float).reshape(-1)
        if x_0_hat.size != self.n_x:
            raise ValueError("The arrival state has an incompatible shape.")
        if not np.all(np.isfinite(x_0_hat)):
            raise ValueError("The arrival state must contain finite values.")

        if self.problem is None or self.problem_needs_rebuild:
            self._build_optimization_problem()

        # Expanding the quadratic arrival, process, and measurement costs
        # leaves a constant block-tridiagonal Hessian. Only q depends on the
        # current arrival state, inputs, and measurements.
        input_effect = self.B @ self.U
        self.linear_cost.fill(0.0)
        self.linear_cost[0] -= 2.0 * self.Sigma_0 @ x_0_hat
        self.linear_cost[:-1] += (
            2.0 * self.A_transpose_Q @ input_effect
        ).T
        self.linear_cost[1:] -= (2.0 * self.Q_w @ input_effect).T
        self.linear_cost[1:] -= (
            2.0 * self.C_transpose_R @ self.Y
        ).T
        self.linear_cost *= self.state_scale[None, :]
        self.problem.update(q=self.linear_cost.reshape(-1))
    
    def estimate(self):
        """
        Solve the optimization problem to estimate the states.

        Returns:
            numpy.ndarray: Estimated states over the horizon.
        """
        if self.problem is None:
            raise RuntimeError("Call setup_optimization_problem before estimate.")

        if self.solution is not None:
            # Shift the previous optimal trajectory by one sample before the
            # next solve. The last state is completed with one model step.
            self.warm_start_solution[:, :-1] = self.solution[:, 1:]
            self.warm_start_solution[:, -1] = (
                self.A @ self.solution[:, -1]
                + self.B @ self.U[:, -1]
            )
            self.problem.warm_start(
                x=(
                    self.warm_start_solution
                    / self.state_scale[:, None]
                ).T.reshape(-1)
            )

        result = self.problem.solve()
        status = result.info.status.lower()
        if not status.startswith("solved"):
            print("The problem was not solved. Status:", result.info.status)
            return None
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError("The solver returned an invalid state estimate.")

        scaled_solution = result.x.reshape(self.N + 1, self.n_x).T
        self.solution = self.state_scale[:, None] * scaled_solution
        return self.solution.copy()
        
