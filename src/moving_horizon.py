import cvxpy as cp
import numpy as np

class MovingHorizonEstimator():
    def __init__(self, A_matrix, B_matric, C_matrix, N):
        """
        Initialize the Moving Horizon Estimator.

        Args:
            A_matrix (numpy.ndarray): System dynamics matrix.
            B_matrix (numpy.ndarray): Input matrix.
            C_matrix (numpy.ndarray): Measurement matrix.
            N (int): Estimation horizon length.
        """
        # System dimensions
        self.n_x = A_matrix.shape[0] # Number of state observables
        self.n_u = B_matric.shape[1] # Number of inputs
        self.n_y = C_matrix.shape[0] # Number of measurements

        # System Matrices        
        self.A = A_matrix
        self.B = B_matric
        self.C = C_matrix
        
        self.N = N
        self.Q = np.eye(self.n_x)       # Process noise weight
        self.R = np.eye(self.n_y)       # Measurement noise weight
        self.Sigma_0 = np.eye(self.n_x) # Initial state error covariance
        
        # Initialize state and measurement history
        self.U = np.zeros((self.n_u, N), dtype=np.float64)
        self.Y = np.zeros((self.n_y, N), dtype=np.float64)

        # Initialize optimization variables and problem
        self.X = cp.Variable((self.n_x, N+1))
        self.objective = None
        self.constraints = []
        self.problem = None
        
    def update_measurements(self, u_new, y_new):
        """
        Update the measurement and input history with new data.

        Args:
            u_new (numpy.ndarray): New control input.
            y_new (numpy.ndarray): New measurement.
        """
        # Roll the matrices to discard the oldest data and append new data
        self.U = np.roll(self.U, -1, axis=1)
        self.U[:, -1] = u_new
        self.Y = np.roll(self.Y, -1, axis=1)
        self.Y[:, -1] = y_new
        
    def setup_optimization_problem(self, x_0_hat):
        """
        Setup the optimization problem for the estimator.

        Args:
            x_0_hat (numpy.ndarray): Initial state estimate.
        """
        # Reset objective and constraints for new optimization setup
        self.objective = cp.quad_form(self.X[:, 0] - x_0_hat, self.Sigma_0)
        self.constraints = []

        # Add objective terms and constraints for each time step in the horizon
        for k in range(self.N):
            self.objective += cp.quad_form(self.Y[:, k] - self.C @ self.X[:, k], self.R)
            self.objective += cp.quad_form(self.X[:, k+1] - self.A @ self.X[:, k] - self.B @ self.U[:, k], self.Q)

        # Define the optimization problem
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)    
    
    def estimate(self):
        """
        Solve the optimization problem to estimate the states.

        Returns:
            numpy.ndarray: Estimated states over the horizon.
        """
        self.problem.solve()
        return self.X.value
    
    
if __name__ == "__main__":
    # Example usage
    A = np.array([[1, 1], [0, 1]])  # System dynamics matrix
    B = np.array([[0.5], [1]])  # Input matrix
    C = np.array([[1, 0]])  # Measurement matrix
    Q = np.eye(2)  # Process noise covariance
    R = np.eye(1)  # Measurement noise covariance
    P_0 = np.eye(2)  # Initial state covariance
    N = 5  # Horizon length

    mhe = MovingHorizonEstimator(A, B, C, N)
    x_0_hat = np.array([0, 0])  # Initial state estimate

    # Setup optimization problem
    mhe.setup_optimization_problem(x_0_hat)

    # Simulate some measurements and control inputs
    for _ in range(N):
        u_new = np.random.randn(1)  # New control input
        y_new = np.random.randn(1)  # New measurement
        mhe.update_measurements(u_new, y_new)

    # Estimate the states
    x_hat = mhe.estimate()
    print(x_hat)