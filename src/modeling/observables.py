import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures


class Observables:
    """Factory for observable functions used to build Koopman embeddings.

    Provides nonlinear feature, radial-basis, tensor, and time-delay lifts.
    """

    
    @staticmethod
    def identity():
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
        poly = PolynomialFeatures(degree, include_bias=False)
        
        def poly_function(x):
            # PolynomialFeatures expects samples in rows.
            return poly.fit_transform(x.T).T

        return poly_function
    
    @staticmethod
    def exp(rate=1.0):
        return lambda x: np.exp(rate * x)
    
    @staticmethod
    def random_fourier_features(n_states, n_features, kernel_width=1.0):
        """
        Create a function to compute Random Fourier Features for a given input using the specified number of features
        and kernel width.

        Parameters:
            n_states (int): The dimensionality of the state space.
            n_features (int): The number of random Fourier features to generate.
            kernel_width (float): The kernel width parameter for the Gaussian kernel approximation.

        Returns:
            function: A function that computes the Random Fourier Features for a given input.
        """
        if n_states < 1 or n_features < 1 or kernel_width <= 0.0:
            raise ValueError("Feature dimensions and kernel_width must be positive.")

        # Sample random weights and offsets for the Fourier features
        random_weights = np.random.normal(0, 1 / kernel_width, (n_states, n_features))
        random_offsets = np.random.uniform(0, 2 * np.pi, n_features)
        
        def fourier_features_function(x):
            """
            Computes the Random Fourier Features for the input x.

            Args:
                x (numpy.ndarray): Input data with shape (n_states, n_samples).

            Returns:
                numpy.ndarray: Features with shape (n_features, n_samples).
            """
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[0] != n_states:
                raise ValueError("x has an incompatible shape.")
            if not np.all(np.isfinite(x)):
                raise ValueError("x must contain finite values.")

            # Project the input data using the random weights and add the offsets
            projection = np.dot(random_weights.T, x) + random_offsets[:, np.newaxis]
            # Compute the cosine transformation and scale the features
            features = np.sqrt(2 / n_features) * np.cos(projection)
            return features

        return fourier_features_function
        
    @staticmethod
    def rbf(state, num_centers, centers=None, random_centers=False, rbf_type="gauss",  gamma=1.0, polyharmonic_coeff=1.0):
        """
        Create a radial basis function (RBF) using the given number of centers and gamma value.

        Parameters:
        centers (numpy.ndarray): The centers of the RBF.
        gamma (float): The gamma parameter for the RBF.

        Returns:
        function: A function that computes the RBF values for a given input.
        """
        state = np.asarray(state, dtype=float)
        if state.ndim != 2:
            raise ValueError("state must have shape (n_states, n_samples).")
        state_dim = state.shape[0]
        if num_centers < 1:
            raise ValueError("num_centers must be positive.")
        if centers is None and not random_centers and num_centers > state.shape[1]:
            raise ValueError("num_centers cannot exceed the number of state samples.")
        
        # Choose the centers
        if centers is not None:
            centers = np.asarray(centers, dtype=float)
            if centers.shape != (state_dim, num_centers):
                raise ValueError("centers has an incompatible shape.")
        elif random_centers:
            # Choose the centers randomly
            lower = np.min(state, axis=1)
            upper = np.max(state, axis=1)
            centers = np.random.uniform(lower[:, None], upper[:, None],
                                        size=(state_dim, num_centers))
        else:
            # Choose the centers by K-Means clustering
            # Reshape the state_data to (num_samples, state_dim)
            state_reshaped = np.transpose(state)
            
            # Initialize the k-means model
            kmeans = KMeans(n_clusters=num_centers, random_state=42, n_init=10)
            
            # Fit the k-means model to your data
            kmeans.fit(state_reshaped)
            
            # Get the cluster centers
            centers = kmeans.cluster_centers_.T

        def rbf_function(x):
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[0] != state_dim:
                raise ValueError("x has an incompatible shape.")
            if not np.all(np.isfinite(x)):
                raise ValueError("x must contain finite values.")

            # Compute the squared distance in a vectorized manner
            diff = x[:, :, np.newaxis] - centers[:, np.newaxis, :]
            r_squared = np.sum(diff ** 2, axis=0)
            
            # Compute RBF values
            if(rbf_type == "gauss"):
                rbf_values = np.exp(-gamma * r_squared).T
            elif(rbf_type == "thinplate"):
                radius = np.sqrt(r_squared)
                safe_radius = np.where(radius > 0.0, radius, 1.0)
                rbf_values = (r_squared * np.log(safe_radius)).T
            elif(rbf_type == "polyharmonic"):
                radius = np.sqrt(r_squared)
                safe_radius = np.where(radius > 0.0, radius, 1.0)
                rbf_values = (radius ** polyharmonic_coeff * np.log(safe_radius)).T
            else:
                raise ValueError("This type of RBF basis function is not defined.")
            return rbf_values
        
        return rbf_function
    
    @staticmethod
    def hermite_polynomial(degree):
        """
        Generate a function to compute the Hermite polynomial of a given degree.

        Parameters:
            degree (int): The degree of the Hermite polynomial.

        Returns:
            function: A function that computes the value of the Hermite polynomial of the given degree at a given point.
        """
        if not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        # Coefficients for the Hermite polynomial of the given degree
        coefficients = np.zeros(degree + 1)
        coefficients[degree] = 1  # Set the coefficient of the highest degree term to 1

        def hermite_function(x):
            """
            Compute the Hermite polynomial of the specified degree at point x.

            Args:
                x (float or numpy.ndarray): The input value(s) at which to evaluate the Hermite polynomial.

            Returns:
                float or numpy.ndarray: The value of the Hermite polynomial at x.
            """
            return np.polynomial.hermite.hermval(x, coefficients)

        return hermite_function
    
    @staticmethod
    def tensor_based(state, order=2):
        """
        Create a tensor-based function as an observable using the given order.

        Parameters:
            state (numpy.ndarray): The state vector of the system.
            order (int): The order of the tensor-based function.

        Returns:
            function: A function that computes the tensor-based values for a given input.
        """
        state = np.asarray(state, dtype=float)
        if state.ndim != 2:
            raise ValueError("state must have shape (n_states, n_samples).")
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer.")
        state_dim = state.shape[0]

        def tensor_function(x):
            x = np.asarray(x, dtype=float)
            if x.ndim != 2 or x.shape[0] != state_dim:
                raise ValueError("x has an incompatible shape.")

            # Start with all ordered monomials of degree one.
            tensor_features = [x.copy()]
            current_order = x.copy()
            
            # Construct higher-order terms
            for _ in range(1, order):
                current_order = np.array([
                    current_order[i] * x[j]
                    for i in range(current_order.shape[0])
                    for j in range(state_dim)
                ])
                tensor_features.append(current_order)
            
            return np.vstack(tensor_features)
        
        return tensor_function
    
    @staticmethod
    def time_delay_observables(delay, embedding_dim):
        if delay < 1 or embedding_dim < 1:
            raise ValueError("delay and embedding_dim must be positive.")

        def observable(x):
            x = np.asarray(x, dtype=float)
            if x.ndim != 2:
                raise ValueError("x must have shape (n_states, n_samples).")
            num_states, num_samples = x.shape
            # Adjust the output size to ensure it has the same number of samples
            delayed_x = np.zeros((num_states * embedding_dim, num_samples))
            for i in range(embedding_dim):
                for j in range(num_states):
                    # Shift the data with respect to the delay and embedding dimension
                    shifted_data = np.roll(x[j, :], i*delay)
                    # Use zeros for initial values that cannot be filled due to the delay
                    if i*delay > 0:
                        shifted_data[:i*delay] = 0
                    delayed_x[j + i*num_states, :] = shifted_data
            return delayed_x
        return observable
