import numpy as np


class Perceptron:
    def __init__(self, feature_dim, num_classes):
        """
        in this constructor you have to initialize the weights of the model with zeros. Do not forget to put 
        the bias term! 
        """
        self.weights = np.zeros([feature_dim+1, num_classes])
        
    def train(self, feature_vector, y):
        """
        this function gets a single training feature vector (feature_vector) with its label (y) and adjusts 
        the weights of the model with perceptron algorithm. 
        Hint: use self.predict() in your implementation.
        """
        y_star = self.predict(feature_vector)
#         print(y_star.shape)
#         print(y_star.shape)
#         print(self.weights[y_star].shape)
#         print(feature_vector.shape)
        
        if y_star != y:
            self.weights[:,y_star] -= feature_vector
            self.weights[:,y] += feature_vector

    def predict(self, feature_vector):
        """
        returns the predicted class (y-hat) for a single instance (feature vector).
        Hint: use np.argmax().
        """
        return np.argmax(feature_vector.dot(self.weights))