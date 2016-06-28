
# Template by Bruce Maxwell
# Spring 2015
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np
import scipy.cluster.vq as vq


class Classifier:
    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type

    def type(self, newtype=None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def confusion_matrix(self, truecats, classcats):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.

        '''
        truecats = truecats.T.tolist()[0]
        classcats = classcats.T.tolist()[0]

        unique_true, mappings_true = np.unique(truecats,return_inverse=True)
        unique_class, mappings_class = np.unique(classcats,return_inverse=True)

        conf_matrix = np.matrix(np.zeros((len(unique_true),len(unique_true))))

        for i in range(len(classcats)):
            conf_matrix[classcats[i],truecats[i]] += 1

        return conf_matrix

    def confusion_matrix_str(self, cmtx):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        s = 'Confusion Matrix\n'
        s += '          '
        for i in range(cmtx.shape[0]):
            s += " TrueC%d" % i

        l = np.matrix(["PredC%d" % i for i in range(cmtx.shape[0])]).T
        cmtx = np.hstack((l,cmtx))
        s += "\n"
        for i in range(cmtx.shape[0]):
            s += str((cmtx[i,:].tolist()[0])).strip('[]') +"\n"

        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)


class NaiveBayes(Classifier):
    '''NaiveBayes implements a simple NaiveBayes classifier using a
    Gaussian distribution as the pdf.

    '''

    def __init__(self, dataObj=None, headers=[], categories=None):
        '''Takes in a Data object with N points, a set of F headers, and a
        matrix of categories, one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'Naive Bayes Classifier')

        # store the headers used for classification
        self.headers = headers

        # number of classes and number of features
        self.num_features = len(headers)
        self.num_classes = None

        # original class labels
        self.categories = categories

        # unique data for the Naive Bayes: means, variances, scales
        self.means = []
        self.variances = []
        self.scales = []

        if dataObj is not None:
            A = dataObj.get_data(headers)
            self.build(A, categories)

    def build(self, A, categories):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        unique, mapping = np.unique(np.array(categories.T), return_inverse=True)

        self.num_classes = len(unique)
        self.num_features = A.shape[1]
        self.categories = categories

        # create the matrices for the means, vars, and scales
        self.means = np.zeros((self.num_classes, self.num_features))
        self.variances = np.zeros((self.num_classes, self.num_features))
        self.scales = np.zeros((self.num_classes, self.num_features))

        for i in range(self.num_classes):
            self.means[i, :] = np.mean(A[(mapping == i), :], axis=0)
            self.variances[i, :] = np.var(A[(mapping == i), :], axis=0)
            self.scales[i, :] = 1 / (np.sqrt(2 * np.pi * self.variances[i, :]))

        return

    def classify(self, A, return_likelihoods=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_likelihoods
        is True, it also returns the NxC likelihood matrix.

        '''

        # error check to see if A has the same number of columns as
        # the class means
        if A.shape[1] != self.means.shape[1]:
            print "Error"
            return

        # make a matrix that is N x C to store the probability of each
        # class for each data point
        P = np.matrix(np.zeros((A.shape[0], self.num_classes)))

        # calculate the probabilities by looping over the classes
        #  with numpy-fu you can do this in one line inside a for loop
        for i in range(self.num_classes):
            P[:,i] = np.prod(np.multiply(self.scales[i,:], np.exp(-(np.square(A - self.means[i,:]))/(2*self.variances[i,:]))),axis=1)


        # calculate the most likely class for each data point
        cats = np.argmax(P, axis=1)  # take the argmax of P along axis 1

        # use the class ID as a lookup to generate the original labels
        labels = self.categories[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.means[i, :]) + "\n"
            s += 'Var   : ' + str(self.variances[i, :]) + "\n"
            s += 'Scales: ' + str(self.scales[i, :]) + "\n"

        s += "\n"
        return s

    def write(self, filename):
        '''Writes the Bayes classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the Bayes classifier from the file'''
        # extension
        return


class KNN(Classifier):
    def __init__(self, dataObj=None, headers=[], categories=None, K=None):
        '''Take in a Data object with N points, a set of F headers, and a
        matrix of categories, with one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')

        # store the headers used for classification
        self.headers = headers

        # number of classes and number of features
        self.num_features = len(headers)
        self.num_classes = 0

        # original class labels
        self.categories = categories

        # unique data for the KNN classifier: list of exemplars (matrices)
        self.exemplars = []
        if dataObj is not None:
            #store data obj
            self.data_obj = dataObj
            A = dataObj.get_data(headers)
            self.build(A,categories)

    def build(self, A, categories, K=None):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        unique, mapping = np.unique(np.array(categories.T), return_inverse=True)
        self.num_classes = len(unique)
        self.num_features = A.shape[0]
        self.categories = categories

        # for each category i, build the set of exemplars
        for i in range(self.num_classes):
            if K is None:
                self.exemplars.append(A[(mapping == i),:])
            else:
                codebook,codes = vq.kmeans(A[(mapping == i),:],K)
                self.exemplars.append(codebook)

        return

    def classify(self, A, K=3, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.

        The parameter K specifies how many neighbors to use in the
        distance computation. The default is three.'''

        # error check to see if A has the same number of columns as the class means

        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.matrix(np.zeros((A.shape[0],self.num_classes))) # a matrix of zeros that is N (rows of A) x C (number of classes)

        print "Classifying using KNN"

        for i in range(self.num_classes):
            temp = np.matrix(np.zeros((A.shape[0],self.exemplars[i].shape[0])))
            for j in range(self.exemplars[i].shape[0]):
                # for k in range(self.exemplars[i].shape[0]):
                #     temp[j,k] = np.linalg.norm(A[j,:]-self.exemplars[i][k,:])
                temp[:,j] = np.sum(np.square(A - self.exemplars[i][j,:]),axis=1)

            temp = np.sort(temp,axis=1)
            D[:,i] = np.sum(temp[:,K],axis=1)

        print D

        # for each class i
        # make a temporary matrix that is N x M where M is the number of examplars (rows in exemplars[i])
        # calculate the distance from each point in A to each point in exemplar matrix i (for loop)
        # sort the distances by row
        # sum the first K columns
        # this is the distance to the first class

        # calculate the most likely class for each data point
        cats = np.argmin(D,axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.categories[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s

    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        # extension
        return
