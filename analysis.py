import random
import sys

__author__ = "Yashaswi Mohanty"
__email__ = "ymohanty@colby.edu"
__version__ = "2/21/2016"

import numpy as np
import scipy.stats
import scipy.cluster.vq as vq
import data
import math
import scipy.spatial.distance as norms
import pandas


# Takes in a list of column headers and the Data object and returns a list of 2-element
# lists with the minimum and maximum values for each column.
def data_range(data_obj, column_headers):
    range_list = []
    columns = data_obj.get_data(column_headers).transpose().tolist()
    for column in columns:
        min_max_list = [max(column), min(column)]
        range_list.append(min_max_list)
    return range_list


# Takes in a list of column headers and the Data object and returns a list of the mean values for each column
def mean(data_obj, column_headers):
    mean_list = []
    columns = data_obj.get_data(column_headers).transpose().tolist()
    for column in columns:
        mean_list.append(np.mean(column))
    return mean_list


# Takes in a list of column headers and the Data object and returns a list of the standard deviation
# for each specified column
def stdev(data_obj, column_headers):
    stdev_list = []
    columns = data_obj.get_data(column_headers).transpose().tolist()
    for column in columns:
        stdev_list.append(np.std(column))
    return stdev_list


# Takes in a list of column headers and the Data object and returns a list of the median
# for each specified column
def median(data_obj, column_headers):
    median_list = []
    columns = data_obj.get_data(column_headers).tolist()
    for column in columns:
        median_list.append(np.median(column))
    return median_list


# Takes in a list of column headers and the Data object and returns a matrix with each column normalized
# so its minimum value is mapped to zero and its maximum value is mapped to 1
def normalize_columns_separately(data_obj, column_headers):
    final_columns = []
    # print column_headers
    columns = data_obj.get_data(column_headers).transpose().tolist()
    for column in columns:
        temp_column = []
        max_num = max(column)
        min_num = min(column)
        for number in column:
            number -= min_num
            number *= 1 / (max_num - min_num)
            temp_column.append(number)
        final_columns.append(temp_column)
    # print "Normalized matrix"
    # print np.matrix(final_columns).transpose()
    print "\n\n"
    return np.matrix(final_columns).transpose()


# Takes in a list of column headers and the Data object and returns a matrix with each entry normalized so that the
# minimum value (of all the data in this set of columns)
# is mapped to zero and its maximum value is mapped to 1.
def normalize_columns_together(data_obj, column_headers):
    final_columns = []
    columns = data_obj.get_data(column_headers).T.tolist()
    max_num = max(columns[0])
    min_num = min(columns[0])
    for column in columns:
        max_num = max(max_num, max(column))
        min_num = min(min_num, min(column))
    # print (max_num,min_num)
    for column in columns:
        temp_column = []
        for number in column:
            number -= min_num
            number *= 1 / (max_num - min_num)
            temp_column.append(number)
        final_columns.append(temp_column)
    print np.matrix(final_columns).transpose()
    return np.matrix(final_columns).transpose()


def pca(d, headers, normalize=True):
    if normalize:
        A = normalize_columns_separately(d, headers)

        m = []
        for i in range(A.shape[1]):
            m.append(np.mean(A[:, i]))
    else:
        A = d.get_data(headers)

        # mean values of the columns of A
        m = np.matrix(mean(d, headers))

    # the difference matrix
    D = A - m

    # singular value decomposition
    U, S, V = np.linalg.svd(D, full_matrices=False)

    # get eigenvalues
    evals = []
    for i in range(len(S)):
        evals.append((math.pow(S[i], 2)) / (A.shape[0] - 1))

    evals = np.matrix(evals)

    # generate projected data
    pdata = (V * D.T).T

    return data.PCAData(headers, pdata, evals, V, m)


# generate a linear regression between the independent and dependent variables in a data set
def linear_regression(data_obj, ind, dep):
    """

    :param data_obj: Data object associated with a data set
    :param ind: list of independent variable headers
    :param dep: dependent variable header
    :return: tuple containing regression data
    """
    y = data_obj.get_data([dep])
    A = data_obj.get_data(ind)

    A = np.append(A, np.ones((A.shape[0], 1)), axis=1)

    #    The matrix A.T * A is the covariance matrix of the independent
    #    data, and we will use it for computing the standard error of the
    #    linear regression fit below
    AAinv = np.linalg.inv(np.dot(A.T, A))

    #    This solves the equation y = Ab, where A is a matrix of the
    #    independent data, b is the set of unknowns as a column vector,
    #    and y is the dependent column of data.  The return value x
    #    contains the solution for b and a bunch of other data
    x = np.linalg.lstsq(A, y)
    b = x[0]
    N = y.shape[0]  # number of rows in y
    C = len(b)  # number of roys in b

    df_e = N - C  # number of degrees of freedom of the error
    df_r = C - 1  # number of degrees of freedom of the model fit

    error = y - np.dot(A, b)  # error term of the standard model
    # [a-b for a,b in zip(y,np.dot(A,y))]

    # the sum squared error, which is the sum of the
    #    squares of the errors computed in the prior step, divided by the
    #    number of degrees of freedom of the error
    sse = np.dot(error.T, error) / df_e

    stderr = np.sqrt(np.diagonal(sse[0, 0] * AAinv))

    t = b.T / stderr

    p = 2 * (1 - scipy.stats.t.cdf(abs(t), df_e))

    r2 = 1 - error.var() / y.var()

    return b, sse, r2, t, p


def save_analysis(filename, datafile, b, sse, r2, t, p, ind, dep):
    """

    :param dep: The dependent variable
    :param ind: The list of independent variables.
    :param datafile: The data file used for the regression
    :param filename:The name of the file in which the analysis would be stored
    :param b: the beta values computed
    :param sse: Sum of errors squared
    :param r2: Coefficient indicating fit quality
    :param t: T statistic given by betas divided by standard error
    :param p: probability of a random relationship
    """

    f = open(filename + '.txt', mode='w')
    f.write("***Linear Regression Report***\n")
    f.write("File: %s\n" % datafile)
    f.write("Independent variables: ")
    for i in range(len(ind)):
        f.write(ind[i] + "\t")
    f.write("\nDependent variable: %s\n" % dep)
    f.write("Beta Coefficients: \n")
    for i in range(len(b)):
        f.write('\t\tB%d: %f\n' % (len(b) - 1 - i, b[i, 0]))

    f.writelines(
        ["Sum Squared Error: %f\n" % sse[0, 0], "R^2: %f\n" % r2, 'T-Statistic: %s\n' % t[0], 'P-value: %s\n' % p[0]])


'''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes, and representation error.
    '''


def kmeans_numpy(d, headers, K, whiten=True):
    # assign to A the result of getting the data from your Data object
    A = d.get_data(headers)

    # assign to W the result of calling vq.whiten on A
    W = vq.whiten(A)

    # assign to codebook, bookerror the result of calling vq.kmeans with W and K
    codebook, bookerror = vq.kmeans(W, K)

    # assign to codes, error the result of calling vq.vq with W and the codebook
    codes, error = vq.vq(W, codebook)

    # return codebook, codes, and error
    return codebook, codes, error


# prep the k-means clustering algorithm by getting initial cluster means
def kmeans_init(d, K, categories=None):
    # list of means to be returned
    means = []

    # data matrix
    A = d

    N = A.shape[0]

    if categories is None:
        for i in range(K):
            means.append(A[np.random.randint(0, N)].tolist()[0])
        # print means

    else:
        if K != max(categories) + 1:
            print "The highest category label and specified clusters should be the same"
            return
        for i in range(K):
            sum = np.zeros(A.shape[1])  # will store the sum of vectors belonging to a category.
            num_elem = 0  # counts the number of elements in a category
            for j in range(len(categories)):
                if categories[j] == i:
                    sum = np.add(sum, A[j].tolist()[0])
                    num_elem += 1
            sum = 1 / float(num_elem) * sum
            means.append(sum)

    return np.matrix(means)


# classify points by distance from mean.
def kmeans_classify(A, means, metric):
    # set up the lists to return
    data_classes = []
    data_metrics = []

    # set up the distance to be the max number possible
    dist = sys.maxint
    for v in A:  # for every data vector
        index = 0
        for i in range(len(means.tolist())):
            m = means.tolist()[i]
            norm_matrix = np.vstack((v, m))
            if norms.pdist(norm_matrix, metric)[0] < dist:
                dist = norms.pdist(norm_matrix, metric)[0]
                index = i

        data_classes.append([index])
        data_metrics.append([dist])
        dist = sys.maxint

    return np.matrix(data_classes), np.matrix(data_metrics)


def kmeans_algorithm(A, means, metric):
    # set up some useful constants
    MIN_CHANGE = 1e-7
    MAX_ITERATIONS = 100
    D = means.shape[1]
    K = means.shape[0]
    N = A.shape[0]

    # iterate no more than MAX_ITERATIONS
    for i in range(MAX_ITERATIONS):
        # calculate the codes
        codes, errors = kmeans_classify(A, means, metric)

        # calculate the new means
        newmeans = np.zeros_like(means)
        counts = np.zeros((K, 1))
        for j in range(N):
            newmeans[codes[j, 0], :] += A[j, :]
            counts[codes[j, 0], 0] += 1.0

        # finish calculating the means, taking into account possible zero counts
        for j in range(K):
            if counts[j, 0] > 0.0:
                newmeans[j, :] /= counts[j, 0]
            else:
                newmeans[j, :] = A[random.randint(0, A.shape[0]), :]

        # test if the change is small enough
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call classify with the final means
    codes, errors = kmeans_classify(A, means, metric)

    # return the means, codes, and errors
    return (means, codes, errors)


def kmeans(d, headers, K, metric, whiten=True, categories=None):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes and representation errors.
    If given an Nx1 matrix of categories, it uses the category labels
    to calculate the initial cluster means.
    '''

    # assign to A the result getting the data given the headers
    try:
        A = d.get_data(headers)
    except AttributeError:
        A = d

    if whiten:
        W = vq.whiten(A)
    else:
        W = A

    codebook = kmeans_init(W, K, categories)

    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook
    codebook, codes, errors = kmeans_algorithm(W, codebook, metric)

    # return the codebook, codes, and representation error
    return codebook, codes, errors


# test function
def test_lin_reg(filename, ind, dep):
    b, sse, r2, t, p = linear_regression(data.Data(filename), ind, dep)
    savefilename = filename.split('.')[0] + "-analysis"
    save_analysis(savefilename, filename, b, sse, r2, t, p, ind, dep)







if __name__ == '__main__':
    # test_lin_reg("data-good.csv", ['X0', 'X1'], 'Y')
    # test_lin_reg("data-clean.csv", ['X0', 'X1'], 'Y')
    # test_lin_reg("data-noisy.csv", ['X0', 'X1'], 'Y')
    #
    # test_lin_reg("cars.csv", ['Weight', 'Horsepower'], 'MPG')

    d = data.Data("clusterdata.csv")
    means = kmeans_init(d, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
    kmeans_classify(d, means)
