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
        # System dimensions
        self.n_x = A_matrix.shape[0] # Number of states
        self.n_u = B_matric.shape[1] # Number of inputs
        self.n_y = C_matrix.shape[0] # Number of measurements

        # System Matrices        
        self.A = A_matrix
        self.B = B_matric
        self.C = C_matrix

        self.x_pre = np.zeros((self.n_x), dtype=np.float64)
        self.x_post = np.zeros((self.n_x), dtype=np.float64)
        
        self.Q = np.diag((1e-5) * np.ones([self.n_x]))
        self.R = np.diag((1e+5) * np.ones([self.n_y]))
        self.Sigma_pre = np.zeros((self.n_x, self.n_x), dtype=np.float64)
        self.Sigma_post = np.zeros((self.n_x, self.n_x), dtype=np.float64)
    
    def init_filter(self, x0):
        self.x_post = x0
    
    def prediction_step(self, u):
        self.x_pre = np.dot(self.A, self.x_post) + np.dot(self.B, u)
        self.Sigma_pre = np.dot(self.A, np.dot(self.Sigma_post, self.A.T)) + self.Q
    
    def update_step(self, y):
        y_est = np.dot(self.C, self.x_pre)
        
        # Innovation covariance
        S = np.dot(self.C, np.dot(self.Sigma_pre, self.C.T)) + self.R
        
        # Kalman gain
        K = np.dot(self.Sigma_pre, np.dot(self.C.T, np.linalg.inv(S)))
        
        self.x_post = self.x_pre + np.dot(K, (y - y_est))
        self.Sigma_post = self.Sigma_pre - np.dot(K, np.dot(self.C, self.Sigma_pre))
    
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
        return self.x_post
    

