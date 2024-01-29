import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Observables:
    
    @staticmethod
    def linear():
        return lambda x: x
    
    @staticmethod
    def sin():
        return lambda x: np.sin(x)
    
    @staticmethod
    def cos():
        return lambda x: np.cos(x)
    
    @staticmethod
    def polynomial(degree=2):
        def poly_function(x):
            return np.power(x, degree)
        return poly_function
    
    @staticmethod
    def full_polynomial(degree=2):
        poly = PolynomialFeatures(degree)
        
        def poly_function(x):
            return poly.fit_transform(x)

        return poly_function
    
    @staticmethod
    def exp(rate=1.0):
        return lambda x: np.exp(rate * x)
    
    @staticmethod
    def rbf(n_states, n_centers, centers=None, rbf_type="gauss",  gamma=1.0):
        """
        Create a radial basis function (RBF) using the given number of centers and gamma value.

        Parameters:
        centers (numpy.ndarray): The centers of the RBF.
        gamma (float): The gamma parameter for the RBF.

        Returns:
        function: A function that computes the RBF values for a given input.
        """
        if centers is None:
            centers = np.random.rand(n_states, n_centers)
        
        def rbf_function(x):
            # Compute the squared distance in a vectorized manner
            diff = x[:, :, np.newaxis] - centers[:, np.newaxis, :]
            r_squared = np.sum(diff ** 2, axis=0)
            
            # Compute RBF values
            if(rbf_type == "gauss"):
                rbf_values = np.exp(-gamma * r_squared).T
            elif(rbf_type == "thinplate"):
                rbf_values = (r_squared * np.log(np.sqrt(r_squared))).T
            return rbf_values
        
        return rbf_function