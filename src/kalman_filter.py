import numpy as np

class Kalman_Filter:
    def __init__(self, koopman_model, c_matrix):
        """
        Kalman filter class to be used with the learned model with Koopmanoperator.
        
        Args:
            Psi_x_pre: priori estimated mean of state-observables, (state-observables: projected states into the high dimensional space).
            Psi_x_post: posteriori estimated mean of state-observables.
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            Sigma: Error covariance matrix.
        """
        self.obs_state_dim = koopman_model.shape[0]
        self.measurement_dim  =c_matrix.shape[0]
        
        self.A = koopman_model[:, :self.obs_state_dim]
        self.B = koopman_model[:, self.obs_state_dim:]
        self.C = c_matrix
        
        self.Psi_x_pre = np.zeros((self.obs_state_dim), dtype=np.float64)
        self.Psi_x_post = np.zeros((self.obs_state_dim), dtype=np.float64)
        
        self.Q = np.diag((1e-1) * np.ones([self.obs_state_dim]))
        self.R = np.diag((1e-2) * np.ones([self.measurement_dim]))
        self.Sigma_pre = np.zeros((self.obs_state_dim, self.obs_state_dim), dtype=np.float64)
        self.Sigma_post = np.zeros((self.obs_state_dim, self.obs_state_dim), dtype=np.float64)
    
    def init_filter(self, Psi_x0):
        self.Psi_x_post = Psi_x0
    
    def prediction_step(self, u):
        self.Psi_x_pre = np.dot(self.A, self.Psi_x_post) + np.dot(self.B, u)
        self.Sigma_pre = np.dot(self.A, np.dot(self.Sigma_post, self.A.T)) + self.Q
    
    def update_step(self, y):
        y_est = np.dot(self.C, self.Psi_x_pre)
        
        # Innovation covariance
        S = np.dot(self.C, np.dot(self.Sigma_pre, self.C.T)) + self.R
        
        # Kalman gain
        K = np.dot(self.Sigma_pre, np.dot(self.C.T, np.linalg.inv(S)))
        
        self.Psi_x_post = self.Psi_x_pre + np.dot(K, (y - y_est))
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
        return self.Psi_x_post
    

