import numpy as np
import matplotlib.pyplot as plt
from koopman import EDMDC
from observables import Observables
from kalman_filter import Kalman_Filter

if __name__ == "__main__":
    # Read the data.
    path = "/home/khorshidi/git/koopman/data/"
    X = np.loadtxt(path+"x_no_noise_train.dat", delimiter='\t', dtype=np.float64)
    U = np.loadtxt(path+"u_no_noise_train.dat", delimiter='\t', dtype=np.float64)
    Y = np.loadtxt(path+"y_no_noise_train.dat", delimiter='\t', dtype=np.float64)
    
    # ---------------------------- DMDC ---------------------------- #
    # koopman_model = DMDC(X, U=U, svd_rank=15)
    # koopman_model.fit_model()
    # x_pred = koopman_model.simulate_model(X[:, 0], U, U.shape[1])
    
    # --------------------------- EDMDC ---------------------------- #
    # Create observables
    linear_obs = Observables.linear()
    polynomial_obs = Observables.polynomial(degree=2)
    exp_obs = Observables.exp(rate=0.1)
    sin_obs = Observables.sin()
    rbf_obs = Observables.rbf(n_states=X.shape[0], n_centers=10, rbf_type="thinplate")
    obs_functions =[linear_obs, exp_obs]
    
    # Instantiate the model
    koopman_model = EDMDC(X, U=U, Y=Y, list_of_obs=obs_functions)
    
    # Fit the regression to the data
    koopman_model.fit_model()
    
    # Simulate the *Training* data with learned koopman operator
    x0 = X[:, 0]
    n_sim_steps = U.shape[1]
    x_pred, y_pred = koopman_model.simulate_model(x0, U, n_sim_steps)
    
    # ---------- Plot the results for training data ---------- #
    plt.figure("Position")
    koopman_model.plot(X[:3, :], x_pred[:3, :], "Sim_data", "Koopman", "Base_Position", 0.7)
    
    plt.figure("Orientation")
    koopman_model.plot(X[3:6, :], x_pred[3:6, :], "Sim_data", "Koopman", "Base_Orientation", 0.0002)
    
    plt.figure("Velocity")
    koopman_model.plot(X[6:, :], x_pred[6:, :], "Sim_data", "Koopman", "Base_Velocity", 0.1)
    
    # plt.figure("Output")
    # koopman_model.plot(Y, y_pred, "Sim_data", "Koopman", "Output", 0.1)
    
    # ---------- Simulate the *Test* data with learned koopman operator ---------- #
    X_test = np.loadtxt(path+"x_noise_test.dat", delimiter='\t', dtype=np.float64)
    U_test = np.loadtxt(path+"u_noise_test.dat", delimiter='\t', dtype=np.float64)
    Y_test = np.loadtxt(path+"y_noise_test.dat", delimiter='\t', dtype=np.float64)
    
    # x0 = X_test[:, 0]
    n_sim_steps = U_test.shape[1]
    # x_pred_test, y_pred_test = koopman_model.simulate_model(x0, U_test, n_sim_steps)
    
    # ---------- Kalman Filtering with learned koopman model on Test data ---------- #
    koopman_kalman_filter = Kalman_Filter(koopman_model.Koopman_operator, koopman_model.C_matrix)
    x_est_kkf = koopman_model.state_estimation(koopman_kalman_filter, x0, U_test, Y_test, n_sim_steps)
    
    # Plot the results for training data
    plt.figure("Position_Test")
    koopman_model.plot(X_test[:3, :], x_est_kkf[:3, :], "Sim_data", "Koopman", "Base_Position", 0.7)
    
    plt.figure("Orientation_Test")
    koopman_model.plot(X_test[3:6, :], x_est_kkf[3:6, :], "Sim_data", "Koopman", "Base_Orientation", 0.0002)
    
    plt.figure("Velocity_Test")
    koopman_model.plot(X_test[6:, :], x_est_kkf[6:, :], "Sim_data", "Koopman", "Base_Velocity", 0.1)
    
    plt.show()