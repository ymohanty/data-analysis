import tkMessageBox

__author__ = "Yashaswi Mohanty"
__email__ = "ymohanty@colby.edu"
__version__ = "2/16/2016"

import numpy as np
import csv
import analysis
import xlrd
import time


class Data:
    # constructor
    def __init__(self, filename=None):
        self.raw_data = []
        self.raw_headers = []
        self.raw_types = []
        self.header2raw = {}
        self.matrix_data = np.matrix([])
        self.header2matrix = {}
        self.enum_dicts = {}

        if filename is not None:
            try:

                if filename.split('.')[1] == "csv":
                    self.read_csv(filename)
                elif filename.split('.')[1] == "xls":
                    self.read_xls(filename)

                else:
                    return
            except IndexError:
               # tkMessageBox.showerror(title="Unknown Error", message="An unknown error occured. Please try again.")
               print "something fucked up"

    # opens and reads a csv file
    def read_csv(self, filename):
        f = file(filename, 'rU')
        cread = csv.reader(f)
        self.raw_headers = cread.next()  # read in headers
        self.raw_types = cread.next()  # read in types

        # add raw data to field
        for line in cread:
            self.raw_data.append(line)

        # bind keys to values for header2raw
        for i in range(len(self.raw_headers)):
            self.header2raw[self.raw_headers[i]] = i

        # set up dictionary mappings for enumerated types
        temp_data_list = np.matrix(self.raw_data).transpose().tolist()
        for i in range(len(self.raw_types)):
            if self.raw_types[i] == "enum":
                enum2numeric = {}
                j = -1
                for name in temp_data_list[i]:
                    if name not in enum2numeric.keys():
                        enum2numeric[name] = j + 1
                        j += 1
                self.enum_dicts[self.raw_headers[i]] = enum2numeric

        # bind keys to values for header2matrix
        j = 0
        for i in range(len(self.raw_types)):
            if self.raw_types[i] == "numeric" or self.raw_types[i] == "date" or self.raw_types[i] == "enum":
                self.header2matrix[self.raw_headers[i]] = j
                j += 1

        temp_matrix = []  # list to hold numeric values temporarily

        # add numeric data to the np matrix
        for row_list in self.raw_data:
            temp_row = []
            for i in range(len(row_list)):
                if self.raw_types[i] == 'numeric':
                   temp_row.append(float(row_list[i]))
                elif self.raw_types[i] == 'date':
                    temp_row.append(time.mktime(time.strptime(row_list[i], '%m/%d/%y')))
                elif self.raw_types[i] == 'enum':
                    temp_row.append(self.enum_dicts[self.raw_headers[i]][row_list[i]])

            temp_matrix.append(temp_row)
        self.matrix_data = np.matrix(temp_matrix)

    # extension: read xls files
    def read_xls(self, filename):
        # open excel workbook
        book = xlrd.open_workbook(filename)

        # get first sheet
        sheet = book.sheet_by_index(0)

        # get raw headers and types
        self.raw_headers = sheet.row_values(0)
        self.raw_types = sheet.row_values(1)

        # get raw data in list
        for i in range(2, sheet.nrows):
            self.raw_data.append(sheet.row_values(i))

        # bind keys to values for header2raw
        for i in range(len(self.raw_headers)):
            self.header2raw[self.raw_headers[i]] = i

        # set up dictionary mappings for enumerated types
        temp_data_list = np.matrix(self.raw_data).transpose().tolist()
        for i in range(len(self.raw_types)):
            if self.raw_types[i] == "enum":
                enum2numeric = {}
                j = 0
                for name in temp_data_list[i]:
                    if name not in enum2numeric.keys():
                        enum2numeric[name] = j + 1
                        j += 1
                self.enum_dicts[self.raw_headers[i]] = enum2numeric

        # bind keys to values for header2matrix
        j = 0
        for i in range(len(self.raw_types)):
            if self.raw_types[i] == "numeric" or self.raw_types[i] == "date" or self.raw_types[i] == "enum":
                self.header2matrix[self.raw_headers[i]] = j
                j += 1

        temp_matrix = []  # list to hold numeric values temporarily

        # add numeric data to the np matrix
        for row_list in self.raw_data:
            temp_row = []
            for i in range(len(row_list)):
                if self.raw_types[i] == 'numeric':
                    temp_row.append(float(row_list[i]))
                elif self.raw_types[i] == 'date':
                    temp_row.append(time.mktime(time.strptime(row_list[i], '%m/%d/%y')))
                elif self.raw_types[i] == 'enum':
                    temp_row.append(self.enum_dicts[self.raw_headers[i]][row_list[i]])

            temp_matrix.append(temp_row)
        self.matrix_data = np.matrix(temp_matrix)

    # accessor for raw_headers
    def get_raw_headers(self):
        return self.raw_headers

    # accessor for raw_types
    def get_raw_types(self):
        return self.raw_types

    # returns the number of columns in the data 
    def get_raw_num_columns(self):
        return len(self.raw_data[0])

    # returns the number of rows in the data
    def get_raw_num_rows(self):
        return len(self.raw_data)

    # returns the index'th row in the data
    def get_raw_row(self, index):
        if index >= len(self.raw_data):
            print "Please specify index between 0 and %d" % (len(self.raw_data) - 1)
        else:
            return self.raw_data[index]

    # returns the element at specified by index(row) and string(column)
    def get_raw_value(self, index, header):
        return self.raw_data[index][self.header2raw[header]]

    # returns the data as a nice string
    def __str__(self):
        print np.matrix(self.raw_data)

    # returns the list of headers with numeric data
    def get_headers(self):
        numeric_headers = []
        for i in range(len(self.raw_headers)):
            if self.raw_types[i] == "numeric" or self.raw_types[i] =="enum":
                numeric_headers.append(self.raw_headers[i])
        return numeric_headers

    # returns the number of columns in the numeric data
    def get_num_columns(self):
        return len(self.get_headers())

    # take a row index and return a row of numeric data
    def get_row(self, index):
        return self.matrix_data[index]

    # takes a row index (int) and column header (string) and returns the data in the numeric matrix.
    def get_value(self, index, header):
        return self.matrix_data[index, self.header2matrix[header]]

    # takes a list of columns headers and return a matrix with the data for the columns specified by the headers
    def get_data(self, headers):
        temp_list = []
        temp_matrix = []
        for header in headers:
            # print header
            try:
                # print self.header2matrix[header]
                temp_list.append(self.header2matrix[header])
            except KeyError as k:
                print "*******Key Error******\n\n"
                print k.message
                return
        for i in temp_list:
            # print self.matrix_data.transpose()[i]
            temp_matrix.append(self.matrix_data.transpose().tolist()[i])

        # print "*******TEMP MATRIX********"
        # print temp_matrix
        # print np.matrix(temp_matrix).T

        return np.matrix(temp_matrix).T

    # adds a column to the matrix

    def add_column(self, header, type, column):
        if len(column) != len(self.raw_data):
            print "Incompatible dimensions"
            return
        self.raw_headers.append(header)
        self.raw_types.append(type)
        self.header2raw[header] = len(self.raw_headers) - 1

        # append column to raw data
        transposed_raw = np.matrix(self.raw_data).T.tolist()
        transposed_raw.append(column)
        self.raw_data = np.matrix(transposed_raw).T.tolist()

        if type == "numeric":
            self.header2matrix[header] = len(self.header2matrix)
            transposed_matrix = self.matrix_data.T.tolist()
            transposed_matrix.append(column)
            self.matrix_data = np.matrix(transposed_matrix).T

    def write_to_file(self, filename, headers):
        f = open(filename + ".csv", mode='w')

        # write headers
        for header in headers[:-1]:
            f.write(header + ",")
        f.write(headers[-1] + "\n")

        # write types
        for header in headers[:-1]:
            f.write(self.raw_types[self.header2raw[header]] + ",")
        f.write(self.raw_types[self.header2raw[headers[-1]]] + "\n")

        # write data:
        for i in range(len(self.raw_data)):
            for header in headers[:-1]:
                f.write(str(self.raw_data[i][self.header2raw[header]]) + ",")
            f.write(str(self.raw_data[i][self.header2raw[headers[-1]]]) + "\n")



class PCAData(Data):
    def __init__(self, headers, pdata, evals, evecs, means):
        self.eigenvectors = evecs
        self.eigenvalues = evals
        self.mean_data_values = means
        self.projected_data = pdata
        self.orig_headers = headers
        self.raw_types = ['numeric'] * len(headers)
        self.raw_data = pdata.tolist()
        self.matrix_data = pdata
        self.header2raw = {}

        self.raw_headers = []

        for i in range(len(headers)):
            self.raw_headers.append("PO%d" % i)

        # populate the header2raw dictionary
        for i in range(len(self.raw_headers)):
            self.header2raw[self.raw_headers[i]] = i

        self.header2matrix = self.header2raw

    # returns a copy of the eigenvalues as a single row numpy matrix
    def get_eigenvalues(self):
        return self.eigenvalues

    # returns a copy of the eigenvectors as a numpy matrix with the eigenvectors as rows.
    def get_eigenvectors(self):
        return self.eigenvectors

    # returns the means for each column in the original data as a single row numpy matrix.
    def get_data_means(self):
        return self.mean_data_values

    # returns a copy of the list of the headers from the original data used to generate the projected data.
    def get_data_headers(self):
        return self.orig_headers

    # returns a copy of the matrix data
    def get_matrix_data(self):
        return self.matrix_data


# test
def main():
    d = Data('cars.csv')
    print "Raw Headers"
    print d.get_raw_headers()
    print "\n\n"
    print "Raw number of columns"
    print d.get_raw_num_columns()
    print "\n\n"
    print "Raw number of rows"
    print d.get_raw_num_rows()
    print "\n\n"
    print "13th row"
    print d.get_raw_row(13)
    print "\n\n"
    print "Value at row 6, header 'Car'"
    print d.get_raw_value(6, 'Car')
    print "\n\n"
    print "Matrix data"
    print d.matrix_data
    print "\n\n"
    print "Headers"
    print d.get_headers()
    print "\n\n"
    print "Number of cols"
    print d.get_num_columns()
    print "\n\n"
    print "5th row"
    print d.get_row(5)
    print "\n\n"
    print "Get value"
    print d.get_value(5, 'Horsepower')
    print "\n\n"
    print "get_data function"
    print d.get_data(['Origin', 'Horsepower'])
    print "\n\n"
    print "data range"
    print analysis.data_range(d, ['Origin', 'Horsepower'])
    print "\n\n"
    print "mean of horsepower and origin"
    print analysis.mean(d, ['Horsepower', 'Origin'])
    print "\n\n"
    print "standard deviation for horsepower and origin"
    print analysis.stdev(d, ['Horsepower', 'Origin'])
    print "\n"
    print "normalized columns origin and horsepower"
    print analysis.normalize_columns_separately(d, ['Origin', 'Horsepower'])
    print "\n\n"
    print "normalized together origin and horsepower"
    print analysis.normalize_columns_together(d, ['Origin', 'Horsepower'])
    print "\n\n"
    print "median of columns origin, horspower and weight"
    print analysis.median(d, ['Origin', 'Horsepower', 'Weight'])
    print d.get_data(['Origin', 'Horsepower']).shape


if __name__ == '__main__':
    main()











    #d.write_to_file("test_write_file",d.get_headers())

