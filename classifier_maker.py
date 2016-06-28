__author__ = "Yashaswi Mohanty"
__version__ = "4/18/2016"

import data
import classifiers
import sys
import numpy as np

def read_training_data(training_file, training_labels_file=None):
    if training_labels_file is None:
        d = data.Data(training_file)
        training_cats = d.get_data([d.get_headers()[-1]])
        training_data = d.get_data(d.get_headers()[:-1])
    else:
        d = data.Data(training_file)
        l = data.Data(training_labels_file)
        training_cats = l.get_data(l.get_headers())
        training_data = d.get_data(d.get_headers())

    return training_data, training_cats, d

def read_testing_data(testing_file, testing_labels_file=None):
    if testing_labels_file is None:
        d = data.Data(testing_file)
        testing_cats = d.get_data([d.get_headers()[-1]])
        testing_data = d.get_data(d.get_headers()[:-1])
    else:
        d = data.Data(testing_file)
        l = data.Data(testing_labels_file)
        testing_cats = l.get_data(l.get_headers())
        testing_data = d.get_data(d.get_headers())

    return testing_data, testing_cats, d

def build_classifier(training_data, training_labels, method):
    if method == "Naive Bayes" or method == "naivebayes":
        nbc = classifiers.NaiveBayes()
        nbc.build(training_data,training_labels)
        return nbc
    elif method == "K-Nearest Neighbors" or method == "knn":
        knn = classifiers.KNN()
        knn.build(training_data,training_labels)
        return knn
    else:
        print "Uknown method: Use 'knn' or 'naivebayes'"
        exit(-1)

def main(argv):
    if len(argv) < 4:
        print 'Usage: python %s <classification method> <train data file> <test data file> <optional train categories> <optional test categories>' % (argv[0])
        exit(-1)

    elif len(argv) > 4:
        print "Reading data..."
        training_data,training_labels, dOb_train = read_training_data(argv[2],argv[4])
        testing_data, testing_labels, dObj_test = read_testing_data(argv[3],argv[5])

    else:
        training_data, training_labels, dOb_train = read_training_data(argv[2])
        testing_data, testing_labels, dObj_test = read_testing_data(argv[3])

    print "Building the Classifier..."

    classifier = build_classifier(training_data,training_labels,argv[1])

    print "Classifying test and training data..."

    ctraincats, ctrainlabels = classifier.classify(training_data)
    ctestcats, ctestlabels = classifier.classify(testing_data)

    # recast labels to [0-C-1]
    unique1, mapping1 = np.unique(training_labels.T.tolist()[0],return_inverse=True)
    unique2, mapping2 = np.unique(testing_labels.T.tolist()[0], return_inverse=True)

    mapping1 = np.matrix(mapping1).T
    mapping2 = np.matrix(mapping2).T

    print "Constructing the Confusion matrices"

    cmtx_train = classifier.confusion_matrix(mapping1,ctraincats)
    cmtx_test = classifier.confusion_matrix(mapping2,ctestcats)

    print cmtx_train
    print cmtx_test

    print "\nTraining Data"
    print classifier.confusion_matrix_str(cmtx_train)
    print "\nTesting Data"
    print classifier.confusion_matrix_str(cmtx_test)

    print "Writing to file"

    dObj_test.add_column("class","numeric",ctestcats.T.tolist()[0])

    dObj_test.write_to_file(argv[3].split('.')[0] + "-" + argv[1] + "-classified",dObj_test.get_headers())

if __name__ == '__main__':
    main(sys.argv)









        



