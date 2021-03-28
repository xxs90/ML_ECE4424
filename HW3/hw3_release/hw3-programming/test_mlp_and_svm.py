"""Test class that trains each model type on one or more dataset and evaluates performance."""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from mlp import mlp_train, mlp_predict, logistic, nll, mlp_objective
from kernelsvm import kernel_svm_train, kernel_svm_predict
from scipy.io import loadmat
from plotutils import plot_data, plot_surface
from scipy.optimize import check_grad, approx_fprime
import copy


def objective_wrapper(x, data, labels, params):
    """
    Wrapper for mlp_objective that takes in a vector and reshapes it into the MLP weights structure
    :param x: weights squeezed into a single vector
    :type x: array
    :param data: ndarray of shape (2, n), where each column is a data example
    :type data: ndarray
    :param labels: length-n array of labels in {+1, -1}
    :type labels: array
    :param params: dict containing the MLP options
    :type params: dict
    :return: tuple containing (1) the scalar loss objective value and (2) the gradient vector
    :rtype: tuple
    """
    # test derivative of mlp objective
    num_hidden_units = params['num_hidden_units']
    input_dim = data.shape[0]

    index = 0
    model = dict()
    model['weights'] = list()
    # create input layer
    curr_matrix_size = num_hidden_units[0] * (input_dim + 1)
    model['weights'].append(x[index:index + curr_matrix_size].reshape((num_hidden_units[0], input_dim + 1)))
    index += curr_matrix_size
    # create intermediate layers
    for layer in range(1, len(num_hidden_units)):
        curr_matrix_size = num_hidden_units[layer] * num_hidden_units[layer - 1]
        model['weights'].append(x[index:index + curr_matrix_size].reshape(
            (num_hidden_units[layer], num_hidden_units[layer - 1])))
        index += curr_matrix_size
    # create output layer
    curr_matrix_size = num_hidden_units[-1]
    model['weights'].append(x[index:].reshape((1, num_hidden_units[-1])))

    model['activation_function'] = params['activation_function']

    obj, gradients = mlp_objective(model, data, labels, params['loss_function'])

    grad_vec = np.concatenate([g.ravel() for g in gradients])

    return obj, grad_vec


class MlpSvmTest(unittest.TestCase):
    """Test class that trains each model type on one or more dataset and evaluates performance."""
    def setUp(self):
        """load synthetic binary-class data from MATLAB data file"""

        variables = dict()
        loadmat('syntheticData.mat', variables)

        # use some list comprehensions to clean up MATLAB data conversion
        self.train_labels = [vector[0].ravel() for vector in variables['trainLabels']]
        self.train_data = [matrix[0] for matrix in variables['trainData']]
        self.test_labels = [vector[0].ravel() for vector in variables['testLabels']]
        self.test_data = [matrix[0] for matrix in variables['testData']]

    def test_mlp(self):
        """
        Train 3-layer MLP on datasets 0 to 9. Tests will try multiple random initialization trials to
        reduce chance of bad initialization.
        """
        threshold_accuracy = [0.95,0.9,0.85,0.9,0.9,0.875,0.9,0.875,0.85,0.85] # threshold accuracy values to beat for every dataset
        pass_vec = np.zeros(10, dtype =bool) # stores whether test passed on a dataset or not
        for i in range(10):

            num_hidden_units = [4, 5]
    
            params = {
                'max_iter': 400,
                'activation_function': logistic,
                'loss_function': nll,
                'num_hidden_units': num_hidden_units,
                'lambda': 0.01
            }
    
            input_dim = self.train_data[i].shape[0] + 1
            total_weight_length = input_dim * num_hidden_units[0]
            for j in range(len(num_hidden_units) - 1):
                total_weight_length += num_hidden_units[j] * num_hidden_units[j + 1]
            total_weight_length += num_hidden_units[-1]
    
            # try at most 10 random initializations
            best_accuracy = 0
            for trial in range(10):
                mlp_model = mlp_train(self.train_data[i], self.train_labels[i], params)
                predictions, _, _, _ = mlp_predict(self.test_data[i], mlp_model)
                accuracy = np.mean(predictions == self.test_labels[i])
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    
            print("On dataset %d, 3-layer MLP had test accuracy %2.3f (should be greater than %2.3f)" %
                  (i, best_accuracy,threshold_accuracy[i]))
            pass_vec = best_accuracy - threshold_accuracy[i] < 0
    
        assert np.sum(pass_vec) == 0, "3-layer MLP accuracy was less than threshold for one of the datasets. Could be a bug, or could be bad luck. " \
                      "Try running again to check."
                      
    def test_poly_svm(self):
        """
        Train quadratic polynomial SVM on datasets 0 to 9.
        """
        
        threshold_accuracy = [0.95,0.9,0.85,0.9,0.9,0.875,0.9,0.85,0.85,0.85]  # threshold accuracy values to beat for every dataset
        pass_vec = np.zeros(10, dtype =bool) # stores whether test passed on a dataset or not
        
        for i in range(10):

            params = {'kernel': 'polynomial', 'C': 1.0, 'order': 2}
    
            svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
            predictions, _ = kernel_svm_predict(self.test_data[i], svm_model)
            test_accuracy = np.mean(predictions == self.test_labels[i])

            print("On dataset %d, Polynomial SVM had test accuracy %2.3f (should be greater than %2.3f)" %
                  (i, test_accuracy,threshold_accuracy[i]))
            pass_vec = test_accuracy - threshold_accuracy[i] < 0
            
        assert np.sum(pass_vec) == 0, "Polynomial SVM accuracy was less than threshold for one of the datasets."

    def test_rbf_svm(self):
        """
        Train RBF SVM on datasets 0 to 9.
        """

        threshold_accuracy = [0.95,0.9,0.875,0.95,0.9,0.9,0.95,0.9,0.875,0.875] # threshold accuracy values to beat for every dataset
        pass_vec = np.zeros(10, dtype =bool) # stores whether test passed on a dataset or not
        
        for i in range(10):

            params = {'kernel': 'rbf', 'C': 1.0, 'sigma': 0.5}
    
            svm_model = kernel_svm_train(self.train_data[i], self.train_labels[i], params)
            predictions, _ = kernel_svm_predict(self.test_data[i], svm_model)
            test_accuracy = np.mean(predictions == self.test_labels[i])

            print("On dataset %d, Kernel SVM had test accuracy %2.3f (should be greater than %2.3f)" %
                  (i, test_accuracy,threshold_accuracy[i]))
            pass_vec = test_accuracy - threshold_accuracy[i] < 0
            
        assert np.sum(pass_vec) == 0, "Kernel SVM accuracy was less than threshold for one of the datasets."

if __name__ == '__main__':
    unittest.main()
