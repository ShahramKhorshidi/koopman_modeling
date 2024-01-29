import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class BaseKoopman:
    def __init__(self, X, X_prime=None):
        """_summary_

        Args:
            X (_type_): _description_
            X_prime (_type_, optional): _description_. Defaults to None.
        """
        if X_prime is None:
            self.X = X[:, :-1]
            self.X_prime = X[:, 1:]
        else:
            self.X = X
            self.X_prime = X_prime
            
        self.n_x = self.X.shape[0]
        self.n_samples = self.X.shape[1]
        print("Shape of X, X': (state_dim, n_samples)", self.X.shape)

    def plot(self, x, y, x_legend, y_legend, title, y_lim=0.5):
        t1 = np.arange(x.shape[1])/1000
        t2 = np.arange(y.shape[1])/1000
        string = "XYZ"
        for i in range(3):
            plt.subplot(int("31" + str(i + 1)))
            plt.plot(t1, x[i, :], "b", label=x_legend, linewidth=0.75)
            plt.plot(t2, y[i, :], "r--", label=y_legend, linewidth=0.75)
            plt.ylabel("_" + string[i] + "_")
            plt.ylim((-y_lim,y_lim))
            plt.grid()
        plt.legend(loc="upper right", shadow=True, fontsize="large")
        
        # Create a custom formatter to display integer values on the time axis
        time_formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x)}")
        plt.gca().xaxis.set_major_formatter(time_formatter)
        
        plt.xlabel("time(sec)")
        plt.suptitle(title)


class DMD(BaseKoopman):
    def __init__(self, X, X_prime=None, svd_rank=None):
        """_summary_

        Args:
            X (_type_): _description_
            X_prime (_type_, optional): _description_. Defaults to None.
            svd_rank (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(X, X_prime)
        self.svd_rank = svd_rank
    
    def fit_model(self):
        # Perform Singular Value Decomposition (SVD) on Transposed state
        U, Sigma, VT = np.linalg.svd(self.X.T, full_matrices=False)
        
        # Truncate SVD components if num_singular_values is specified
        if self.svd_rank is not None and 0 < self.svd_rank < len(Sigma):
            U = U[:, :self.svd_rank]
            Sigma = Sigma[:self.svd_rank]
            VT = VT[:self.svd_rank, :]
        
        # Compute Koopman operator with truncated components
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        self.Koopman_operator = (self.X_prime @ U @ Sigma_inv @ VT).T
        
        print("koopman operator shape:", self.Koopman_operator.shape)

    def simulate_model(self, x0, n_steps):
        """Simulate an initial state forward in time using the learned koopman model.
        
        Args:
            x0 (_type_): Initial actual state
            n_steps (int, optional): Number of steps to simulate. Defaults to 1.

        Returns:
            _type_: _description_
        """
        x = np.empty((self.n_x, n_steps))
        x[:, 0] = x0
        
        for i in range(1, n_steps):
            # Extracting A from the Koopman operator
            A = self.Koopman_operator.T[:, :self.n_x]
            
            # Compute the next state
            x[:, i] = np.dot(A, x[:, i-1]) 
        
        return x


class DMDC(BaseKoopman):
    def __init__(self, X, X_prime=None, U=None ,svd_rank=None):
        super().__init__(X, X_prime)
        if U is None:
            raise ValueError("The value for 'U' is missing.")
        else:
            self.U = U[:, :self.n_samples]
        self.svd_rank = svd_rank

    def fit_model(self):
        # Combine the current state and control input
        B = np.vstack((self.X, self.U))
        
        # Perform Singular Value Decomposition (SVD) on Transposed B
        U, Sigma, VT = np.linalg.svd(B.T, full_matrices=False)
        
        # Truncate SVD components if num_singular_values is specified
        if self.svd_rank is not None and 0 < self.svd_rank < len(Sigma):
            U = U[:, :self.svd_rank]
            Sigma = Sigma[:self.svd_rank]
            VT = VT[:self.svd_rank, :]
        
        # Compute Koopman operator with truncated components
        Sigma_inv = np.linalg.pinv(np.diag(Sigma))
        self.Koopman_operator = (self.X_prime @ U @ Sigma_inv @ VT).T
        
        print("koopman operator shape:", self.Koopman_operator.shape)

    def simulate_model(self, x0, u):
        """_summary_

        Args:
            x0 (_type_): _description_
            u (_type_): Control inputs for each step 
            n_steps (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        n_steps = u.shape[1]
        x = np.empty((self.n_x, n_steps))
        x[:, 0] = x0
        
        for i in range(1, n_steps):
            # Extracting A and B from the Koopman operator
            A = self.Koopman_operator.T[:, :self.n_x]
            B = self.Koopman_operator.T[:, self.n_x:]
            
            # Compute the next state
            x[:, i] = np.dot(A, x[:, i-1]) + np.dot(B, u[:, i-1])
        
        return x


class EDMDC(BaseKoopman):
    def __init__(self, X, X_prime=None, U=None, Y=None, list_of_obs=[]):
        super().__init__(X, X_prime)
        if U is None:
            raise ValueError("The value for 'U' is missing.")
        else:
            self.U = U[:, :self.n_samples]
            self.n_u = self.U.shape[0]
        
        if Y is None:
            self.Y = np.zeros((1, self.n_samples))
            self.n_y = self.Y.shape[0]
        else:
            self.Y = Y[:, :self.n_samples]
            self.n_y = self.Y.shape[0]
        
        self.observable_functions = list_of_obs
        self.n_psi_x = 0
        
        self.A_matrix = np.zeros((self.n_psi_x, self.n_psi_x), dtype=np.float64)
        self.B_matrix = np.zeros((self.n_psi_x, self.n_u), dtype=np.float64)
        self.C_matrix = np.zeros((self.n_y, self.n_psi_x), dtype=np.float64)
        
        print("Shape of U is : (input_dim, n_samples)", self.U.shape)
        print("Shape of Y is : (measurement_dim, n_samples)", self.Y.shape)

    def compute_obs(self, x):
        if x.ndim ==1:
            x=x[:, np.newaxis]
        observable_values = [obs_func(x) for obs_func in self.observable_functions]
        lifted_state = np.concatenate(observable_values)
        if self.check_for_overflow(lifted_state):
            raise FloatingPointError("Overflow in caculating state observables.")
        self.n_psi_x = lifted_state.shape[0]
        return lifted_state
    
    def check_for_overflow(self, arr):
        if np.any(np.isinf(arr)) or np.any(np.isnan(arr)) or np.any(np.abs(arr) > np.finfo(np.float64).max):
            return True
        return False
    
    def normalize_data(self):
        # Normalize the state matrix X
        self.X_avg = np.mean(self.X, axis=1, dtype=np.float64)
        self.X_range = np.max(self.X, axis=1) - np.min(self.X, axis=1)
        self.X_normalized = (self.X - self.X_avg[:, None]) / self.X_range[:, None]
        self.X_prime_normalized = (self.X_prime - self.X_avg[:, None]) / self.X_range[:, None]
        
        # Normalize the input matrix U
        self.U_avg = np.mean(self.U, axis=1, dtype=np.float64)
        self.U_range = np.max(self.U, axis=1) - np.min(self.U, axis=1)
        self.U_normalized = (self.U - self.U_avg[:, None]) / self.U_range[:, None]
        
        # Normalize the measurement matrix Y
        self.Y_avg = np.mean(self.Y, axis=1, dtype=np.float64)
        self.Y_range = np.max(self.Y, axis=1) - np.min(self.Y, axis=1)
        self.Y_normalized = (self.Y - self.Y_avg[:, None]) / self.Y_range[:, None]
    
    def fit_model(self):
        # Aplly EDMDC on normalized X, U and Y matrices
        self.normalize_data()
        
        # Combine observables of normalized state with normalized control input
        Psi_X = self.compute_obs(self.X_normalized)
        
        self.n_psi_x = Psi_X.shape[0] 
        
        Psi_XU = np.vstack((Psi_X, self.U_normalized))
        
        # Next state matrix (observables of normalized state prime)
        Psi_X_prime = self.compute_obs(self.X_prime_normalized)
        
        # --------- Solve for the Koopman operator using Closed_Form solution -------------- #
        self.Koopman_operator = np.dot(Psi_X_prime, np.dot(Psi_XU.T, np.linalg.pinv(np.dot(Psi_XU, Psi_XU.T))))
        self.A_matrix = self.Koopman_operator[:, :self.n_psi_x]
        self.B_matrix = self.Koopman_operator[:, self.n_psi_x:]
        # --------------------------------- Solve with LLSQ -------------------------------- #
        # self.Koopman_operator = np.linalg.lstsq(Psi_XU.T, Psi_X_prime.T, rcond=None)[0].T
        # ---------------------------------------------------------------------------------- #
        
        self.C_matrix = np.dot(self.Y_normalized, np.dot(Psi_X.T, np.linalg.pinv(np.dot(Psi_X, Psi_X.T))))
        print("Koopman_operator shape:", self.Koopman_operator.shape)
    
    def simulate_model(self, x0, u):
        # Normalize the initial state and input 
        x0_normalized = (x0 - self.X_avg) / self.X_range
        
        n_steps = u.shape[1]
        u_avg = np.mean(u, axis=1, dtype=np.float64)
        u_range = np.max(u, axis=1) - np.min(u, axis=1)
        u_normalized = (u - u_avg[:, None]) / u_range[:, None]        
        
        # Calculate the observables of the initial state
        Psi_X = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        # Arrays for collecting data
        state = np.empty((self.n_x, n_steps), dtype=np.float64)
        state[:, 0] = x0
        measurement = np.empty((self.n_y, n_steps), dtype=np.float64)
        
        for i in range(1, n_steps):
            # Predict the observables of next state with koopman operator and current input
            Psi_X = (np.dot(self.A_matrix, Psi_X) + np.dot(self.B_matrix, u_normalized[:, i-1]))
            
            if self.check_for_overflow(Psi_X):
                raise FloatingPointError("Overflow in caculating Psi_X.")
            
            # Extract the current state 
            # IMPORTNAT: the first observable function should be always the Identity_obs
            x_normalized = Psi_X[:self.n_x]
            
            # Calculate the measurement
            y_normalized = np.dot(self.C_matrix, Psi_X)
            
            # Scale back to the actual state and measurement
            state[:, i] = (x_normalized * self.X_range + self.X_avg)
            measurement[:, i] = y_normalized * self.Y_range + self.Y_avg
        return state, measurement
    
    def get_linear_sys_matrices(self):
        return self.A_matrix, self.B_matrix, self.C_matrix
    
    def simulate_model_llsq(self, x0, u):
        # Normalize the initial state and input 
        x0_normalized = (x0 - self.X_avg) / self.X_range
        x0_normalized = x0_normalized[:, np.newaxis]
        
        n_steps = u.shape[1]
        u_avg = np.mean(u, axis=1, dtype=np.float64)
        u_range = np.max(u, axis=1) - np.min(u, axis=1)
        u_normalized = (u - u_avg[:, None]) / u_range[:, None]
        
        # Calculate the observables of initial state
        Psi_X = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        # Arrays for collecting data
        state = np.empty((self.n_x, n_steps), dtype=np.float64)
        state[:, 0] = x0
        measurement = np.empty((self.n_y, n_steps), dtype=np.float64)
        
        for i in range(1, n_steps):
            # Combine observables of current state with the control input
            Psi_XU = np.hstack((Psi_X, u_normalized[:, i-1]))
            
            # Ensure Psi_XU is a column vector
            Psi_XU = Psi_XU.reshape(-1, 1)
            
            # Compute next state observables
            Psi_X = np.dot(self.Koopman_operator, Psi_XU).reshape(self.n_psi_x)
            
            # Calculate the measurement
            y_normalized = np.dot(self.C_matrix, Psi_X)
            
            # Extract only the state part
            x_normalized= Psi_X[:self.n_x].reshape(self.n_x)
            
            # Scale back to the actual state and measurement
            state[:, i] = x_normalized * self.X_range + self.X_avg
            measurement[:, i] = y_normalized * self.Y_range + self.Y_avg
        return state, measurement

    def state_estimation(self, kalman_filter, x0, u, y):
        # Normalize the initial state and input
        x0_normalized = (x0 - self.X_avg) / self.X_range
        
        n_steps = u.shape[1]
        u_avg = np.mean(u, axis=1, dtype=np.float64)
        u_range = np.max(u, axis=1) - np.min(u, axis=1)
        u_normalized = (u - u_avg[:, None]) / u_range[:, None]
        
        # Calculate the observables of the initial state
        Psi_x_pre = self.compute_obs(x0_normalized).reshape(self.n_psi_x)
        
        kalman_filter.init_filter(Psi_x_pre)
        
        # Arrays for collecting data
        state_est = np.empty((self.n_x, n_steps), dtype=np.float64)
        
        for i in range(1, n_steps):
            kalman_filter.update_filter(u_normalized[:, i], y[:,i])
            Psi_est = kalman_filter.get_filter_output()
            
            x_normalized_est = Psi_est[:self.n_x]
            state_est[:, i] = (x_normalized_est * self.X_range + self.X_avg)
            
        return state_est
