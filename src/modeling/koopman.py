"""License BSD-3-Clause
Copyright (c) 2024, University of Bonn.
Author: Shahram Khorshidi
"""

import time
import numpy as np
import cvxpy as cp
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _as_data_matrix(data, name):
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (dimension, n_samples).")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _safe_range(data):
    data_range = np.max(data, axis=1) - np.min(data, axis=1)
    return np.where(data_range > np.finfo(float).eps, data_range, 1.0)


class BaseKoopman:
    """Base class for Koopman models built from state snapshot pairs.

    Validates the data and provides shared serialization and plotting helpers.
    """

    def __init__(self, X, X_prime=None):
        """_summary_

        Args:
            X (_type_): _description_
            X_prime (_type_, optional): _description_. Defaults to None.
        """
        X = _as_data_matrix(X, "X")
        if X_prime is None:
            if X.shape[1] < 2:
                raise ValueError("X must contain at least two samples when X_prime is omitted.")
            self.X = X[:, :-1]
            self.X_prime = X[:, 1:]
        else:
            self.X = X
            self.X_prime = _as_data_matrix(X_prime, "X_prime")
            if self.X_prime.shape != self.X.shape:
                raise ValueError("X and X_prime must have the same shape.")
            
        self.n_x = self.X.shape[0]
        self.n_samples = self.X.shape[1]
        print("Shape of X, X': (state_dim, n_samples)", self.X.shape)

    # Save the Koopman model
    def save_model(self, filename):
        """
        Saves the Koopman model to a file using dill.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Load the Koopman model
    @classmethod
    def load_model(cls, filename):
        """
        Loads a Koopman model from a file using dill.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def plot(self, x, y, x_legend, y_legend, title, y_lim=None):
        t1 = np.arange(x.shape[1])/1000
        t2 = np.arange(y.shape[1])/1000
        string = "XYZ"
        for i in range(3):
            plt.subplot(int("31" + str(i + 1)))
            plt.plot(t1, x[i, :], "b", label=x_legend, linewidth=0.75)
            plt.plot(t2, y[i, :], "r--", label=y_legend, linewidth=0.75)
            plt.ylabel("_" + string[i] + "_")
            if y_lim is not None:
                plt.ylim((-y_lim,y_lim))
            plt.grid()
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        
        # Create a custom formatter to display integer values on the time axis
        time_formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x)}")
        plt.gca().xaxis.set_major_formatter(time_formatter)
        
        plt.xlabel("time(sec)")
        plt.suptitle(title)


class DMD(BaseKoopman):
    """Dynamic Mode Decomposition (DMD) for autonomous systems.

    Fits a linear operator that advances state snapshots without control inputs.
    """

    def __init__(self, X, X_prime=None, svd_rank=None):
        """_summary_

        Args:
            X (_type_): _description_
            X_prime (_type_, optional): _description_. Defaults to None.
            svd_rank (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(X, X_prime)
        if svd_rank is not None and (not isinstance(svd_rank, (int, np.integer)) or svd_rank < 1):
            raise ValueError("svd_rank must be a positive integer or None.")
        self.svd_rank = svd_rank
    
    def fit_model(self):
        # X_prime = A X, with X = U Sigma V.T
        U, Sigma, VT = np.linalg.svd(self.X, full_matrices=False)
        
        # Truncate SVD components if num_singular_values is specified
        if self.svd_rank is not None and 0 < self.svd_rank < len(Sigma):
            U = U[:, :self.svd_rank]
            Sigma = Sigma[:self.svd_rank]
            VT = VT[:self.svd_rank, :]
        
        # Compute the least-squares Koopman operator A = X_prime X^dagger
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        self.Koopman_operator = self.X_prime @ VT.T @ Sigma_inv @ U.T
        
        print("koopman operator shape:", self.Koopman_operator.shape)

    def simulate_model(self, x0, n_steps):
        """Simulate an initial state forward in time using the learned koopman model.
        
        Args:
            x0 (numpy.ndarray): Initial actual state
            n_steps (int): Number of steps to simulate.

        Returns:
            numpy.ndarray: The open loop simulation of the trajectory.
        """
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model before simulate_model.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != self.n_x:
            raise ValueError("The initial state has an incompatible shape.")
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        x = np.empty((self.n_x, n_steps))
        x[:, 0] = x0
        
        for i in range(1, n_steps):
            # Compute the next state
            x[:, i] = np.dot(self.Koopman_operator, x[:, i-1])
        
        return x


class DMDC(BaseKoopman):
    """Dynamic Mode Decomposition with Control (DMDc) for actuated systems.

    Fits linear state and input matrices directly from snapshot data.
    """

    def __init__(self, X, X_prime=None, U=None ,svd_rank=None):
        super().__init__(X, X_prime)
        if U is None:
            raise ValueError("The value for 'U' is missing.")
        U = _as_data_matrix(U, "U")
        if U.shape[1] < self.n_samples:
            raise ValueError("U does not contain enough input samples.")
        self.U = U[:, :self.n_samples]
        if svd_rank is not None and (not isinstance(svd_rank, (int, np.integer)) or svd_rank < 1):
            raise ValueError("svd_rank must be a positive integer or None.")
        self.svd_rank = svd_rank
        self.n_u = self.U.shape[0]
        
        self.A_matrix = np.zeros((self.n_x, self.n_x))
        self.B_matrix = np.zeros((self.n_x, self.n_u))
        self.C_matrix = np.eye(self.n_x)

    def fit_model(self):
        """
        Solve the linear least-square problem with SVD decomposition to find Koopman operator.

        Returns:
            numpy.ndarray: Singular values of SVD decomposition.
        """
        # Combine the current state and control input
        Omega = np.vstack((self.X, self.U))
        
        # X_prime = [A B] Omega, with Omega = U Sigma V.T
        U, Sigma, VT = np.linalg.svd(Omega, full_matrices=False)
        
        # Truncate SVD components if num_singular_values is specified
        if self.svd_rank is not None and 0 < self.svd_rank < len(Sigma):
            U = U[:, :self.svd_rank]
            Sigma = Sigma[:self.svd_rank]
            VT = VT[:self.svd_rank, :]
        
        # Compute [A B] = X_prime Omega^dagger
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        self.Koopman_operator = self.X_prime @ VT.T @ Sigma_inv @ U.T
        print("koopman operator shape:", self.Koopman_operator.shape)
        
        # Extracting A and B from the Koopman operator
        self.A_matrix = self.Koopman_operator[:, :self.n_x]
        self.B_matrix = self.Koopman_operator[:, self.n_x:]
        return Sigma
    
    def optimize_model(self, regularization=0.0):
        """
        Optimize the least-squares problem to find the Koopman operator.
        This is an alternative to fit_model.
        """
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be non-negative.")
        Omega = np.vstack((self.X, self.U))
        
        # Define the variable for koopman operator
        rows_K, cols_K = self.X_prime.shape[0], Omega.shape[0]
        K = cp.Variable((rows_K, cols_K))
        
        # Least squares with optional Tikhonov regularization.
        residual = self.X_prime - K @ Omega
        objective = cp.Minimize(
            cp.sum_squares(residual) + regularization * cp.sum_squares(K)
        )
        
        # Set up the problem
        problem = cp.Problem(objective)
        
        # Solve the problem
        problem.solve()

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Koopman optimization failed with status {problem.status}.")

        self.Koopman_operator = K.value
        print("Status:", problem.status)

        # Extracting A and B from the Koopman operator
        self.A_matrix = self.Koopman_operator[:, :self.n_x]
        self.B_matrix = self.Koopman_operator[:, self.n_x:]
        
    def simulate_model(self, x0, u):
        """_summary_

        Args:
            x0 (_type_): _description_
            u (_type_): Control inputs for each step 
            n_steps (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before simulate_model.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        n_steps = u.shape[1]
        x = np.empty((self.n_x, n_steps))
        x[:, 0] = x0
        
        for i in range(1, n_steps):            
            # Compute the next state
            x[:, i] = np.dot(self.A_matrix, x[:, i-1]) + np.dot(self.B_matrix, u[:, i-1])
        
        return x
    
    def get_linear_sys_matrices(self):
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before requesting matrices.")
        return self.A_matrix, self.B_matrix, self.C_matrix
    
    def kf_estimation(self, kalman_filter, u, y):        
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before state estimation.")
        u = _as_data_matrix(u, "u")
        y = _as_data_matrix(y, "y")
        if u.shape[0] != self.n_u or y.shape[0] != self.n_x:
            raise ValueError("The input or measurement has an incompatible shape.")
        n_steps = y.shape[1]
        if u.shape[1] < n_steps - 1:
            raise ValueError("u must contain an input for every state transition.")
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps))
        state_est[:, 0] = y[:, 0]
        kalman_filter.set_mu_post(state_est[:, 0])

        for i in range(1, n_steps):
            # Run the Kalman Filter
            kalman_filter.update_filter(u[:, i-1], y[:, i])
            x_est = kalman_filter.get_filter_output()
            
            state_est[:, i] = x_est
            
        return state_est
    
    def mhe_estimation(self, mhe_estimator, x_init, u, y):
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before state estimation.")
        u = _as_data_matrix(u, "u")
        y = _as_data_matrix(y, "y")
        if u.shape[0] != self.n_u or y.shape[0] != self.n_x:
            raise ValueError("The input or measurement has an incompatible shape.")
        n_steps = y.shape[1]
        if u.shape[1] < n_steps - 1:
            raise ValueError("u must contain an input for every state transition.")

        x_init = np.asarray(x_init, dtype=float)
        if x_init.ndim == 1:
            arrival_state = x_init.copy()
        elif x_init.ndim == 2:
            arrival_state = x_init[:, 0].copy()
        else:
            raise ValueError("x_init must be a vector or a state trajectory.")
        if arrival_state.size != self.n_x:
            raise ValueError("x_init has an incompatible shape.")
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps))
        state_est[:, 0] = arrival_state
        update_times = []
        for i in range(1, n_steps):
            # Update the MHE with new control input and measurement
            time_now = time.perf_counter()
            mhe_estimator.update_measurements(u[:, i-1], y[:, i])

            if i >= mhe_estimator.get_horizon():  # Start estimation after filling the horizon
                mhe_estimator.setup_optimization_problem(arrival_state)
                
                # Estimate the states
                estimated_states = mhe_estimator.estimate()
                
                # Use the last state in the horizon as the current state estimate
                if estimated_states is not None:
                    state_est[:, i] = estimated_states[:, -1]
                else:
                    raise ValueError("Output is None. Optimization was unsuccessful.")
                
                # The next horizon starts at the second state of this solution.
                arrival_state = estimated_states[:, 1].copy()
                update_times.append(time.perf_counter() - time_now)
                
            else:
                # Fill in the initial state for steps before the horizon is full
                if x_init.ndim == 2 and i < x_init.shape[1]:
                    state_est[:, i] = x_init[:, i]
                else:
                    state_est[:, i] = y[:, i]
        if update_times:
            mean_update_time = 1000.0 * np.mean(update_times)
            print(f"Mean MHE update time: {mean_update_time:.3f} ms")
        return state_est


class EDMDC(BaseKoopman):
    """Extended Dynamic Mode Decomposition with Control (EDMDc).

    Lifts states through configurable observables while retaining direct inputs.
    """

    def __init__(self, X, X_prime=None, U=None, Y=None, list_of_obs=None):
        super().__init__(X, X_prime)
        if U is None:
            raise ValueError("The value for 'U' is missing.")
        U = _as_data_matrix(U, "U")
        if U.shape[1] < self.n_samples:
            raise ValueError("U does not contain enough input samples.")
        self.U = U[:, :self.n_samples]
        self.n_u = self.U.shape[0]
        print("Shape of U is : (input_dim, n_samples)", self.U.shape)
        if Y is None:
            self.Y = None
            self.n_y = self.n_x
        else:
            Y = _as_data_matrix(Y, "Y")
            if Y.shape[1] < self.n_samples:
                raise ValueError("Y does not contain enough measurement samples.")
            self.Y = Y[:, :self.n_samples]
            self.n_y = self.Y.shape[0]
            print("Shape of Y is : (measurement_dim, n_samples)", self.Y.shape)
        
        if not list_of_obs:
            raise ValueError("At least one state observable is required.")
        self.observable_functions = list(list_of_obs)
        self.n_psi_x = 0

    def init_linear_matrices(self):
        self.A_matrix = np.zeros((self.n_psi_x, self.n_psi_x))
        self.B_matrix = np.zeros((self.n_psi_x, self.n_u))
        self.C_matrix = np.zeros((self.n_y, self.n_psi_x))
        if self.Y is None:
            self.C_matrix[:, :self.n_x] = np.eye(self.n_x)
    
    def compute_obs(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.ndim != 2 or x.shape[0] != self.n_x:
            raise ValueError("x has an incompatible shape.")
        observable_values = [obs_func(x) for obs_func in self.observable_functions]
        lifted_state = np.concatenate(observable_values, axis=0)
        if lifted_state.shape[1] != x.shape[1]:
            raise ValueError("Every observable must preserve the number of samples.")
        if self.check_for_overflow(lifted_state):
            raise FloatingPointError("Overflow in caculating state observables.")
        self.n_psi_x = lifted_state.shape[0]
        return lifted_state
    
    def check_for_overflow(self, arr):
        return not np.all(np.isfinite(arr))
    
    def normalize_data(self):
        # Normalize the state matrix X
        self.X_avg = np.mean(self.X, axis=1)
        self.X_range = _safe_range(self.X)
        self.X_normalized = (self.X - self.X_avg[:, None]) / self.X_range[:, None]
        self.X_prime_normalized = (self.X_prime - self.X_avg[:, None]) / self.X_range[:, None]
        
        # Normalize the input matrix U
        self.U_avg = np.mean(self.U, axis=1)
        self.U_range = _safe_range(self.U)
        self.U_normalized = (self.U - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Normalize the measurement matrix Y
        if self.Y is not None:
            self.Y_avg = np.mean(self.Y, axis=1)
            self.Y_range = _safe_range(self.Y)
            self.Y_normalized = (self.Y - self.Y_avg[:, None]) / self.Y_range[:, None]

    def normalize_measurement(self, y):
        y = _as_data_matrix(y, "y")
        if y.shape[0] != self.n_y:
            raise ValueError("y has an incompatible shape.")
        if self.Y is None:
            return (y - self.X_avg[:, None]) / self.X_range[:, None]
        return (y - self.Y_avg[:, None]) / self.Y_range[:, None]
    
    def fit_model(self):
        # Aplly EDMDC on normalized X, U and Y matrices
        self.normalize_data()
        
        # Combine observables of normalized state with normalized control input
        Psi_X = self.compute_obs(self.X_normalized)
        if not np.allclose(Psi_X[:self.n_x], self.X_normalized):
            raise ValueError("The first state observable must be the identity.")
        
        self.n_psi_x = Psi_X.shape[0] 
        
        Psi_XU = np.vstack((Psi_X, self.U_normalized))
        
        # Next state matrix (observables of normalized state prime)
        Psi_X_prime = self.compute_obs(self.X_prime_normalized)
        
        # Initialize the linear dynamics matrices to zero with proper dimension
        self.init_linear_matrices()
        
        # --------- Solve for the Koopman operator using Closed_Form solution -------------- #
        # self.Koopman_operator = np.dot(Psi_X_prime, np.dot(Psi_XU.T, np.linalg.pinv(np.dot(Psi_XU, Psi_XU.T))))
        
        # --------------------------------- Solve with LLSQ -------------------------------- #
        self.Koopman_operator = np.linalg.lstsq(Psi_XU.T, Psi_X_prime.T, rcond=None)[0].T
        
        self.A_matrix = self.Koopman_operator[:, :self.n_psi_x]
        self.B_matrix = self.Koopman_operator[:, self.n_psi_x:]
        if self.Y is not None:
            self.C_matrix = np.linalg.lstsq(
                Psi_X.T, self.Y_normalized.T, rcond=None
            )[0].T
        print("Koopman_operator shape:", self.Koopman_operator.shape)

    def optimize_model(self, regularization=0.0):
        """
        Optimize the least-squares problem to find the Koopman operator.
        This is an alternative to fit_model.
        """
        # Aplly EDMDC on normalized X, U and Y matrices
        self.normalize_data()
        
        # Combine observables of normalized state with normalized control input
        Psi_X = self.compute_obs(self.X_normalized)
        if not np.allclose(Psi_X[:self.n_x], self.X_normalized):
            raise ValueError("The first state observable must be the identity.")
        self.n_psi_x = Psi_X.shape[0] 
        Psi_XU = np.vstack((Psi_X, self.U_normalized))
        
        # Next state matrix (observables of normalized state prime)
        Psi_X_prime = self.compute_obs(self.X_prime_normalized)
        
        # Initialize the linear dynamics matrices to zero with proper dimension
        self.init_linear_matrices()
        
        # Define the variable for koopman operator
        rows_K, cols_K = Psi_X_prime.shape[0], Psi_XU.shape[0]
        K = cp.Variable((rows_K, cols_K))
        
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be non-negative.")
        residual = Psi_X_prime - K @ Psi_XU
        objective = cp.Minimize(
            cp.sum_squares(residual) + regularization * cp.sum_squares(K)
        )
        
        # Set up the problem
        problem = cp.Problem(objective)
        
        # Solve the problem
        problem.solve()

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Koopman optimization failed with status {problem.status}.")

        self.Koopman_operator = K.value
        print("Status:", problem.status)
        print(self.Koopman_operator.shape, "#####")

        # Extracting A and B from the Koopman operator
        self.A_matrix = self.Koopman_operator[:, :self.n_psi_x]
        self.B_matrix = self.Koopman_operator[:, self.n_psi_x:]
        if self.Y is not None:
            self.C_matrix = np.linalg.lstsq(
                Psi_X.T, self.Y_normalized.T, rcond=None
            )[0].T
        
    def simulate_model(self, x0, u):
        # Normalize the initial state and input 
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before simulate_model.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        
        n_steps = u.shape[1]
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Calculate the observables of the initial state
        Psi_X = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        # Arrays for collecting data
        state = np.empty((self.n_x, n_steps))
        state[:, 0] = x0
        # measurement = np.empty((self.n_y, n_steps), dtype=np.float64)
        
        for i in range(1, n_steps):
            # Predict the observables of next state with koopman operator and current input
            Psi_X = (np.dot(self.A_matrix, Psi_X) + np.dot(self.B_matrix, u_normalized[:, i-1]))
            
            if self.check_for_overflow(Psi_X):
                print("index= ", i)
                print("Psi_X = ", Psi_X)
                raise FloatingPointError("Overflow in caculating Psi_X.")
            
            # Extract the current state 
            # IMPORTNAT: the first observable function should be always the Identity_obs
            x_normalized = Psi_X[:self.n_x]
            
            # Calculate the measurement
            # y_normalized = np.dot(self.C_matrix, Psi_X)
            
            # Scale back to the actual state and measurement
            state[:, i] = (x_normalized * self.X_range + self.X_avg)
            # measurement[:, i] = y_normalized * self.Y_range + self.Y_avg
        return state
    
    def get_linear_sys_matrices(self):
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before requesting matrices.")
        return self.A_matrix, self.B_matrix, self.C_matrix
    
    def simulate_model_llsq(self, x0, u):
        # Normalize the initial state and input 
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before simulate_model_llsq.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        x0_normalized = x0_normalized[:, np.newaxis]
        
        n_steps = u.shape[1]
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Calculate the observables of initial state
        Psi_X = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        # Arrays for collecting data
        state = np.empty((self.n_x, n_steps))
        state[:, 0] = x0
        # measurement = np.empty((self.n_y, n_steps), dtype=np.float32)
        
        for i in range(1, n_steps):
            # Combine observables of current state with the control input
            Psi_XU = np.hstack((Psi_X, u_normalized[:, i-1]))
            
            # Ensure Psi_XU is a column vector
            Psi_XU = Psi_XU.reshape(-1, 1)
            
            # Compute next state observables
            Psi_X = np.dot(self.Koopman_operator, Psi_XU).reshape(self.n_psi_x)
            
            # # Calculate the measurement
            # y_normalized = np.dot(self.C_matrix, Psi_X)
            
            # Extract only the state part
            x_normalized= Psi_X[:self.n_x].reshape(self.n_x)
            
            # Scale back to the actual state and measurement
            state[:, i] = x_normalized * self.X_range + self.X_avg
            # measurement[:, i] = y_normalized * self.Y_range + self.Y_avg
        return state

    def kf_estimation(self, kalman_filter, x0, u, y):
        # Normalize the initial state, input and measurement
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before state estimation.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        
        y_normalized = self.normalize_measurement(y)
        n_steps = y_normalized.shape[1]
        if u.shape[1] < n_steps - 1:
            raise ValueError("u must contain an input for every state transition.")
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Calculate the observables of the initial state
        Psi_x_post = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        kalman_filter.set_mu_post(Psi_x_post)
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps))
        state_est[:, 0] = x0
        
        for i in range(1, n_steps):
            # Run the Kalman Filter
            kalman_filter.update_filter(u_normalized[:, i-1], y_normalized[:, i])
            Psi_est = kalman_filter.get_filter_output()
            
            x_normalized_est = Psi_est[:self.n_x]
            state_est[:, i] = (x_normalized_est * self.X_range + self.X_avg)
            
        return state_est
    
    def mhe_estimation(self, mhe_estimator, x0, u, y):
        # Normalize the initial state, input, and measurement
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model or optimize_model before state estimation.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        
        y_normalized = self.normalize_measurement(y)
        n_steps = y_normalized.shape[1]
        if u.shape[1] < n_steps - 1:
            raise ValueError("u must contain an input for every state transition.")
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        arrival_state = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps))
        state_est[:, 0] = x0
        
        for i in range(1, n_steps):
            # Update the MHE with new control input and measurement
            mhe_estimator.update_measurements(
                u_normalized[:, i-1], y_normalized[:, i]
            )
            
            if i >= mhe_estimator.get_horizon():  # Start estimation after filling the horizon
                # Setup optimization problem with the initial state estimate
                mhe_estimator.setup_optimization_problem(arrival_state)
                
                # Estimate the states
                estimated_states = mhe_estimator.estimate()
                if estimated_states is None:
                    raise RuntimeError("The moving-horizon optimization was unsuccessful.")
                
                # Use the last state in the horizon as the current state estimate
                x_normalized_est = estimated_states[:self.n_x, -1]
                state_est[:, i] = (x_normalized_est * self.X_range + self.X_avg)
                arrival_state = estimated_states[:, 1].copy()
            else:
                # Fill in the initial state for steps before the horizon is full
                if self.n_y == self.n_x:
                    state_est[:, i] = y[:, i]
                else:
                    state_est[:, i] = x0
        
        return state_est


class EDMDCKIC(BaseKoopman):
    """EDMDc with Koopman with Inputs and Control (KIC).

    Lifts both states and inputs through configurable observable functions.
    """

    def __init__(self, X, X_prime=None, U=None, Y=None,
                 list_of_state_obs=None, list_of_input_obs=None):
        super().__init__(X, X_prime)
        if U is None:
            raise ValueError("The value for 'U' is missing.")
        U = _as_data_matrix(U, "U")
        if U.shape[1] < self.n_samples:
            raise ValueError("U does not contain enough input samples.")
        self.U = U[:, :self.n_samples]
        self.n_u = self.U.shape[0]
        print("Shape of U is : (input_dim, n_samples)", self.U.shape)
        if Y is None:
            self.Y = None
            self.n_y = self.n_x
        else:
            Y = _as_data_matrix(Y, "Y")
            if Y.shape[1] < self.n_samples:
                raise ValueError("Y does not contain enough measurement samples.")
            self.Y = Y[:, :self.n_samples]
            self.n_y = self.Y.shape[0]
            print("Shape of Y is : (measurement_dim, n_samples)", self.Y.shape)
        
        if not list_of_state_obs or not list_of_input_obs:
            raise ValueError("State and input observables are required.")
        self.state_obs_func = list(list_of_state_obs)
        self.input_obs_func = list(list_of_input_obs)
        self.n_psi_x = 0
        self.n_psi_u = 0

    def init_linear_matrices(self):
        self.A_matrix = np.zeros((self.n_psi_x, self.n_psi_x))
        self.B_matrix = np.zeros((self.n_psi_x, self.n_psi_u))
        self.C_matrix = np.zeros((self.n_y, self.n_psi_x))
        if self.Y is None:
            self.C_matrix[:, :self.n_x] = np.eye(self.n_x)
    
    def get_linear_sys_matrices(self):
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model before requesting matrices.")
        return self.A_matrix, self.B_matrix, self.C_matrix

    def compute_obs(self, x, u=None):
        if x is not None:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x[:, np.newaxis]
            if x.ndim != 2 or x.shape[0] != self.n_x:
                raise ValueError("x has an incompatible shape.")
            state_obs_values = [obs_func(x) for obs_func in self.state_obs_func]
            lifted_state = np.concatenate(state_obs_values, axis=0)
            if lifted_state.shape[1] != x.shape[1]:
                raise ValueError("Every state observable must preserve the number of samples.")
            self.n_psi_x = lifted_state.shape[0]
        else:
            lifted_state = np.array([])        
        
        if u is not None:
            u = np.asarray(u, dtype=float)
            if u.ndim == 1:
                u = u[:, np.newaxis]
            if u.ndim != 2 or u.shape[0] != self.n_u:
                raise ValueError("u has an incompatible shape.")
            input_observable_values = [obs_func(u) for obs_func in self.input_obs_func]
            lifted_input = np.concatenate(input_observable_values, axis=0)
            if lifted_input.shape[1] != u.shape[1]:
                raise ValueError("Every input observable must preserve the number of samples.")
            self.n_psi_u = lifted_input.shape[0]
        else:
            lifted_input = np.array([])
            
        if self.check_for_overflow(lifted_state) or self.check_for_overflow(lifted_input):
            raise FloatingPointError("Overflow in calculating observables.")
        
        return lifted_state, lifted_input
    
    def check_for_overflow(self, arr):
        return not np.all(np.isfinite(arr))
    
    def normalize_data(self):
        # Normalize the state matrix X
        self.X_avg = np.mean(self.X, axis=1)
        self.X_range = _safe_range(self.X)
        self.X_normalized = (self.X - self.X_avg[:, None]) / self.X_range[:, None]
        self.X_prime_normalized = (self.X_prime - self.X_avg[:, None]) / self.X_range[:, None]
        
        # Normalize the input matrix U
        self.U_avg = np.mean(self.U, axis=1)
        self.U_range = _safe_range(self.U)
        self.U_normalized = (self.U - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Normalize the measurement matrix Y
        if self.Y is not None:
            self.Y_avg = np.mean(self.Y, axis=1)
            self.Y_range = _safe_range(self.Y)
            self.Y_normalized = (self.Y - self.Y_avg[:, None]) / self.Y_range[:, None]

    def normalize_measurement(self, y):
        y = _as_data_matrix(y, "y")
        if y.shape[0] != self.n_y:
            raise ValueError("y has an incompatible shape.")
        if self.Y is None:
            return (y - self.X_avg[:, None]) / self.X_range[:, None]
        return (y - self.Y_avg[:, None]) / self.Y_range[:, None]
    
    def fit_model(self, lambda_val=1):
        # Aplly EDMDCKIC on normalized X, U and Y matrices
        if not np.isfinite(lambda_val) or lambda_val < 0.0:
            raise ValueError("lambda_val must be non-negative.")
        self.normalize_data()
        
        # Compute observables for normalized state and input
        Psi_X, Psi_U = self.compute_obs(self.X_normalized, self.U_normalized) 
        if not np.allclose(Psi_X[:self.n_x], self.X_normalized):
            raise ValueError("The first state observable must be the identity.")
        
        # Combine observables of normalized state with normalized input
        Psi_XU = np.vstack((Psi_X, Psi_U))
        
        # Next state matrix (observables of normalized state prime)
        Psi_X_prime, _ = self.compute_obs(self.X_prime_normalized)
        
        # Initialize the linear dynamics matrices to zero with proper dimension
        self.init_linear_matrices()
        
        # Solve the ridge-regression problem without forming normal equations.
        if lambda_val == 0.0:
            regression_matrix = Psi_XU.T
            regression_target = Psi_X_prime.T
        else:
            n_features = Psi_XU.shape[0]
            regression_matrix = np.vstack((
                Psi_XU.T,
                np.sqrt(lambda_val) * np.eye(n_features),
            ))
            regression_target = np.vstack((
                Psi_X_prime.T,
                np.zeros((n_features, Psi_X_prime.shape[0])),
            ))
        self.Koopman_operator = np.linalg.lstsq(
            regression_matrix, regression_target, rcond=None
        )[0].T
        
        self.A_matrix = self.Koopman_operator[:, :self.n_psi_x]
        self.B_matrix = self.Koopman_operator[:, self.n_psi_x:self.n_psi_x+self.n_psi_u]
        if self.Y is not None:
            self.C_matrix = np.linalg.lstsq(
                Psi_X.T, self.Y_normalized.T, rcond=None
            )[0].T
        
        print("Koopman_operator shape:", self.Koopman_operator.shape)

    def simulate_model(self, x0, u, num_steps):
        # Normalize the initial state and input
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model before simulate_model.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        if num_steps < 1 or num_steps > u.shape[1] + 1:
            raise ValueError("num_steps is incompatible with the available inputs.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        x0_normalized = x0_normalized[:, np.newaxis]
        
        n_steps = num_steps
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Arrays for collecting data
        state = np.empty((self.n_x, n_steps))
        state[:, 0] = x0
        
        for i in range(1, n_steps):
            # Compute the observables of the current state
            Psi_X, _ = self.compute_obs(x0_normalized)
        
            # Compute the observables of the current control input
            _, Psi_U = self.compute_obs(None, u_normalized[:, i-1][:, np.newaxis])            
            
            # Combine observables of current state with observables of the control input
            Psi_XU = np.vstack((Psi_X, Psi_U))
            
            if self.check_for_overflow(Psi_XU):
                print("index= ", i)
                print("Psi_X = ", Psi_X)
                raise FloatingPointError("Overflow in caculating Psi_XU.")
            
            # Ensure Psi_XU is a column vector
            Psi_XU = Psi_XU.reshape(-1, 1)
        
            # Compute next state observables
            # Psi_X_next = np.dot(self.Koopman_operator, Psi_XU).reshape(-1)
            Psi_X_next = (np.dot(self.A_matrix, Psi_X) + np.dot(self.B_matrix, Psi_U)).reshape(-1)

            # Extract only the state part and ensure it's a column vector for next iteration
            x0_normalized = Psi_X_next[:self.n_x].reshape(-1, 1)

            # Scale back to the actual state
            state[:, i] = x0_normalized.flatten() * self.X_range + self.X_avg

        return state
    
    def kf_estimation(self, kalman_filter, x0, u, y):
        # Normalize the initial state, input and measurement
        if not hasattr(self, "Koopman_operator"):
            raise RuntimeError("Call fit_model before state estimation.")
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        u = _as_data_matrix(u, "u")
        if x0.size != self.n_x or u.shape[0] != self.n_u:
            raise ValueError("The initial state or input has an incompatible shape.")
        x0_normalized = (x0 - self.X_avg) / self.X_range
        x0_normalized = x0_normalized[:, np.newaxis]
        
        y_normalized = self.normalize_measurement(y)
        n_steps = y_normalized.shape[1]
        if u.shape[1] < n_steps - 1:
            raise ValueError("u must contain an input for every state transition.")
        u_normalized = (u - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps))
        state_est[:, 0] = x0
        Psi_X_post, _ = self.compute_obs(x0_normalized)
        kalman_filter.set_mu_post(Psi_X_post.reshape(self.n_psi_x))
        
        for i in range(1, n_steps):
            # Compute the observables of the current control input
            _, Psi_U = self.compute_obs(None, u_normalized[:, i-1])
            
            # Run the Kalman Filter
            kalman_filter.update_filter(Psi_U.reshape(self.n_psi_u), y_normalized[:, i])
            Psi_est = kalman_filter.get_filter_output()
            
            x_normalized_est = Psi_est[:self.n_x]
            state_est[:, i] = x_normalized_est * self.X_range + self.X_avg
            
        return state_est
