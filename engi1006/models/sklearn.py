################################################################
# HW 5 - ENGI1006
# Nupur Dave
# ncd2123
# This file returns the models based on the users choice. It
# instantiates the class and compares similar results to determine
# the percentage accuracy
###############################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def SKLearnKnnClassifier(training, testing, training_labels, testing_labels, k):
    '''leverage Scikit-learn to implement the k nearest neighbors algorithm
    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array
        k: the number of nearest neighbors to use

    This should instantiate the class from scikit learn, then call
    `.fit` and `.predict`. It should then compare results similar to
    NNClassifier to return a % correct.
    
    See how easy it is to use scikit learn?
    '''
    # instantiate model
    knn = KNeighborsClassifier(k)

    # fit model to training data
    knn.fit(training, training_labels)
    # predict test labels
    knn = knn.predict(testing)

    # return % where prediction matched actual
    percent = sum(knn == testing_labels)/len(testing_labels)
    return percent

def SKLearnSVMClassifier(training, testing, training_labels, testing_labels):
    '''leverage Scikit-learn to implement the support vector machine classifier
    Args:
        training: the subset of data corresponding to the training data as a numpy matrix
        testing:  the subset of data corresponding to the testing data as a numpy matrix
        training_labels: the labels for the training data as a numpy array
        testing_labels: the labels for the testing data as a numpy array

    This should instantiate the class from scikit learn, then call
    `.fit` and `.predict`. It should then compare results similar to
    NNClassifier to return a % correct.
    
    See how easy it is to use scikit learn?
    '''
    # instantiate model
    svm = SVC()

    # fit model to training data
    svm.fit(training, training_labels)

    # predict test labels
    predicted_test_labels = svm.predict(testing)

    # return % where prediction matched actual
    return sum(predicted_test_labels == testing_labels)/len(testing_labels)
