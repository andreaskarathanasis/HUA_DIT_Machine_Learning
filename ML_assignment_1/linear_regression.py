# Import necessary modules
import numpy as np

# Create linear regression class
class LinearRegression:

    # Define init function, set w and b to None by default
    def __init__(self, w=None, b=None) -> None:
        self.w = w
        self.b = b


    # Define fit function with predictors and labels as input
    def fit(self, X, y):

        # Check if X, y inputs are of correct types
        if type(X) is not np.ndarray and type(y) is not np.ndarray:
          raise Exception(f"Expected np.ndarray for X and got: {type(X)}, Expected np.ndarray for y and got: {type(y)}")
        
        # Get X, y dimensions 
        num_rows_X, num_cols_X = X.shape
        num_rows_y = y.size

        # Check for dimension compatibility
        if num_rows_X != num_rows_y:
            raise Exception("Input size doesn't match")

        # Create updated X matrix with last column having every element equal to 1
        ones = np.ones((num_rows_X, 1))
        X = np.append(X, ones, axis=1)

        # Calculate θ parameters through Ordinary Least Squares regression
        theta = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))

        # Save in w attribute all elements of θ vector except from the last 
        self.w = theta[:-1]
        # Save in b attribute the last element of θ vector
        self.b = theta[-1]


    # Define predict function 
    def predict(self, X):

        # Check if model is untrained
        if type(self.w) == None or type(self.b) == None:
            raise Exception("Model is untrained")

        # Calculate predictions
        y = np.dot(X, self.w) + self.b

        # Return predictions
        return y
        

    # Define evaluate function
    def evaluate(self, X, y):

        # Check if model is untrained
        if type(self.w) == None or type(self.b) == None:
            raise Exception("Model is untrained")

        # Call predict function to get predictions for X input
        predictions = self.predict(X)

        # Calculate mean square error using the predictions and the labels
        MSE = np.dot((predictions - y).transpose(), (predictions - y)) / y.size
        # Return predictions and mean square error
        return (predictions, MSE)

    
