"""License BSD-3-Clause
Copyright (c) 2024, University of Bonn.
Author: Shahram Khorshidi
"""

import numpy as np


class Kalman_Filter:
    def __init__(self, A_matrix, B_matric, C_matrix):
        """
        Kalman filter class to be used with the linear model learned with Koopmanoperator.
        
        Args:
            Psi_x_pre: priori estimated mean of state-observables, (state-observables: projected states into the high dimensional space).
            Psi_x_post: posteriori estimated mean of state-observables.
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            Sigma: Error covariance matrix.
        """
        A_matrix = np.asarray(A_matrix, dtype=float)
        B_matric = np.asarray(B_matric, dtype=float)
        C_matrix = np.asarray(C_matrix, dtype=float)
        self._check_system_matrices(A_matrix, B_matric, C_matrix)

        # System dimensions
        self.n_x = A_matrix.shape[0] # Number of states
        self.n_u = B_matric.shape[1] # Number of inputs
        self.n_y = C_matrix.shape[0] # Number of measurements

        # System Matrices        
        self.A = A_matrix
        self.B = B_matric
        self.C = C_matrix

        self.x_pre = np.zeros(self.n_x)
        self.x_post = np.zeros(self.n_x)
        
        self.Q = np.eye(self.n_x)
        self.R = np.eye(self.n_y)      
        self.Sigma_pre = np.eye(self.n_x)
        self.Sigma_post = np.eye(self.n_x)

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

    @staticmethod
    def _check_covariance(matrix, dimension, name):
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain finite values.")
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"{name} has an incompatible shape.")
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"{name} must be symmetric.")
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.min(eigenvalues) < -1e-12:
            raise ValueError(f"{name} must be positive semi-definite.")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return (eigenvectors * eigenvalues) @ eigenvectors.T
    
    def set_mu_post(self, x0):
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != self.n_x:
            raise ValueError("The initial state has an incompatible shape.")
        if not np.all(np.isfinite(x0)):
            raise ValueError("The initial state must contain finite values.")
        self.x_post = x0.copy()

    def set_initial_covariance(self, Sigma):
        self.Sigma_post = self._check_covariance(
            Sigma, self.n_x, "Initial covariance"
        )
        
    def set_process_noise_cov(self, Q):
        self.Q = self._check_covariance(Q, self.n_x, "Process covariance")
        
    def set_measurement_noise_cov(self, R):
        self.R = self._check_covariance(R, self.n_y, "Measurement covariance")
    
    def update_matrices(self, A, B, C):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        C = np.asarray(C, dtype=float)
        self._check_system_matrices(A, B, C)
        if A.shape != self.A.shape or B.shape != self.B.shape or C.shape != self.C.shape:
            raise ValueError("Updated matrices must keep the original dimensions.")
        self.A = A
        self.B = B
        self.C = C
    
    def prediction_step(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != self.n_u:
            raise ValueError("The control input has an incompatible shape.")
        if not np.all(np.isfinite(u)):
            raise ValueError("The control input must contain finite values.")
        self.x_pre = np.dot(self.A, self.x_post) + np.dot(self.B, u)
        self.Sigma_pre = np.dot(self.A, np.dot(self.Sigma_post, self.A.T)) + self.Q
        self.Sigma_pre = 0.5 * (self.Sigma_pre + self.Sigma_pre.T)
    
    def update_step(self, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.n_y:
            raise ValueError("The measurement has an incompatible shape.")
        if not np.all(np.isfinite(y)):
            raise ValueError("The measurement must contain finite values.")
        y_est = np.dot(self.C, self.x_pre)
        
        # Innovation covariance
        S = np.dot(self.C, np.dot(self.Sigma_pre, self.C.T)) + self.R
        
        # Kalman gain
        try:
            K = np.linalg.solve(S, self.C @ self.Sigma_pre).T
        except np.linalg.LinAlgError:
            K = self.Sigma_pre @ self.C.T @ np.linalg.pinv(S)
        
        self.x_post = self.x_pre + np.dot(K, (y - y_est))

        # Joseph form keeps the covariance symmetric and positive semi-definite.
        identity = np.eye(self.n_x)
        correction = identity - K @ self.C
        self.Sigma_post = correction @ self.Sigma_pre @ correction.T + K @ self.R @ K.T
        self.Sigma_post = 0.5 * (self.Sigma_post + self.Sigma_post.T)
    
    def update_filter(self, u, y):
        """_summary_

        Args:
            u (_type_): Control input at the current timestep.
            y (_type_): Measurement at the current timestep.
        """
        self.prediction_step(u)
        self.update_step(y)
    
    def get_filter_output(self):
        """_summary_

        Returns:
            _type_: updated mean of state-observables estimate.
        """
        return self.x_post.copy()
