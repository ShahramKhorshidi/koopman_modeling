import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Obs():
    @staticmethod
    def rbf(centers, rbf_type="gauss", gamma=1.0):
        """
        Create a radial basis function (RBF) using the given centers and gamma value.

        Parameters:
        centers (numpy.ndarray): The centers of the RBF.
        gamma (float): The gamma parameter for the RBF.

        Returns:
        function: A function that computes the RBF values for a given input.
        """
        rbf_type = rbf_type
        def rbf_function(x):
            # Compute the squared distance in a vectorized manner
            diff = x[:, :, np.newaxis] - centers[:, np.newaxis, :]
            r_squared = np.sum(diff ** 2, axis=0)
            
            # Compute RBF values
            if (rbf_type == "gauss"):
                rbf_values = np.exp(-gamma * r_squared).T
            elif(rbf_type == "thinplate"):
                rbf_values = (r_squared * np.log(np.sqrt(r_squared))).T
            return rbf_values

        return rbf_function
    
    @staticmethod
    def compute_obs(x, observable_values):
        observable_values = [obs_func(x) for obs_func in observable_values]
        return np.concatenate(observable_values)
    
    @staticmethod
    def normalize_data(x):
        # Normalize the state matrix X
        x_avg = np.mean(x, axis=1)
        x_range = np.max(x, axis=1) - np.min(x, axis=1)
        x_normalized = (x - x_avg[:, None]) / x_range[:, None]
        return x_normalized
    
    @staticmethod
    def linear():
        return lambda x: x
    
    @staticmethod
    def exp(rate=1.0, clip_range=1000):
        return lambda x: np.exp(rate * np.clip(x, -clip_range, clip_range))
    
    @staticmethod
    def polynomial(degree=2):
        return lambda x: np.power(x, degree)
    
    @staticmethod
    def full_polynomial(degree=2):
        poly = PolynomialFeatures(degree)
        
        def poly_function(x):
            return poly.fit_transform(x)

        return poly_function

if __name__ == "__main__":
    # n_statQ = np.diag((1e-3)) * np.ones([self.obs_state_dim])fted X shape:", lifted_x.shape)
    x = np.array([[1, 2, 3, 4, 5],
                  [0, 2, 8, 4, 6],
                  [9, 8, 4, 2, 1]])
    print("x shape:", x.shape)
    # x_norm = Obs.normalize_data(x)
    
    linear_obs = Obs.linear()
    obs_poly = Obs.full_polynomial()
    # obs_exp = Obs.exp()
    
    # n_center = 5
    # center = np.random.rand(3, n_center)
    # rbf_obs = Obs.rbf(center, rbf_type="thinplate")
    obs_functions = [obs_poly]
    
    lifted_x = Obs.compute_obs(x, obs_functions)
    print(lifted_x.shape)
    # z= np.array([Obs.compute_obs(x[:, i], obs_functions) for i in range(5)]).T
    # print(z)
    
