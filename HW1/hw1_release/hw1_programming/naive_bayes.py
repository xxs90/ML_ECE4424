"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """

    labels = np.unique(train_labels)

    d, n = train_data.shape
    num_classes = labels.size

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES (USING LAPLACE ESTIMATE)

    model = dict()
    class_count = np.zeros((d, num_classes))
    class_total = np.zeros(num_classes)

    for c in range(num_classes):
        class_count[:, c] = train_data[:, train_labels == c].sum(1)
        class_total[c] = np.count_nonzero(train_labels == c)

    prior_prob = np.log(class_total) - np.log(n)
    conditional_prob = np.log(class_count + 1) - np.log(class_total + 2)

    model['prior_prob'] = prior_prob
    model['conditional_prob'] = conditional_prob

    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA

    prior_prob = model['prior_prob']
    conditional_prob = model['conditional_prob']

    p1 = conditional_prob.T.dot(data)
    p2 = np.log(1 - np.exp(conditional_prob)).T.dot(1 - data)
    p3 = prior_prob.reshape((prior_prob.size, 1))
    predict = p1 + p2 + p3

    labels = np.argmax(predict, axis=0)

    return labels
