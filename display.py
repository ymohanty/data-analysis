# Skeleton Tk interface example
# Written by Bruce Maxwell
# Modified by Stephanie Taylor
# Modified by Yashaswi Mohanty (2/14/16)
#
# CS 251
# Spring 2015

import Tkinter as tk
import tkFont as tkf
import math
import random
import tkMessageBox
import scipy.stats
import analysis
import view
import numpy as np
import tkFileDialog
import data as dt
import matplotlib.pyplot as plt
import classifier_maker as cm


# static function returns the distance between two points by using the Euclidean distance formula
def get_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0]) * (loc1[0] - loc2[0]) + (loc1[1] - loc2[1]) * (
        loc1[1] - loc2[1]))  # Euclidean distance formula


# create a class to build and manage the display
# noinspection PyPep8Naming

class DisplayApp:
    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry("%dx%d+50+30" % (self.initDx, self.initDy))

        # set the title of the window
        self.root.title("Data Analysis and Visualization")

        # set the maximum size of the window for resizing
        self.root.maxsize(1600, 900)

        # setup the menus
        self.buildMenus()

        # set up a state variable which controls the number of random data points generated.
        self.num_pts = tk.StringVar(self.root, value=100)

        # manages the distribution to be used while creating the random data points
        self.distribution = ("Uniform", "Uniform")

        # area to check while deleting data points given the size of the objects
        self.size = 5

        # build the controls
        self.buildControls()

        # build the Canvas
        self.buildCanvas()

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print self.canvas.winfo_geometry()

        # set up the key bindings
        self.setBindings()

        # set up the view object
        self.view = view.View()

        # set up the view axes
        self.axes = np.matrix([[0, 0, 0, 1],
                               [1, 0, 0, 1],
                               [0, 0, 0, 1],
                               [0, 1, 0, 1],
                               [0, 0, 0, 1],
                               [0, 0, 1, 1],
                               ])

        # set up graphics lines objects
        self.axis_lines = []

        # set up the application state

        self.objects = []  # list of data objects that will be drawn in the canvas
        self.data = None  # will hold the raw data someday.
        self.filename = None
        self.spatial_data = None  # holds the normalized spatial data to be plotted
        self.color_data = None  # holds the 4th dimension data which is represented through color
        self.size_data = None  # holds the 5th dimension data which is represented through size of ovals.
        self.baseClick = None  # used to keep track of mouse movement
        self.baseClick2 = None  # used to keep track of mouse movement for handling button 3
        self.viewClone = None  # stores clones of the view object
        self.linreg_line_objects = []  # stores graphical objects representing linear regression best fit lines
        self.linreg_endpoints = None  # stores the endpoints of the linear regression best fit line
        self.baseExtent = None  # used to track the original extent of the view volume box during scaling
        self.axis_labels = []  # used to store the axis label objects whose positions are updated as the axes are moved around.
        self.x_label_name = tk.StringVar(self.root, value="x")
        self.y_label_name = tk.StringVar(self.root, value="y")
        self.z_label_name = tk.StringVar(self.root, value="z")
        self.rotation_speed = 5
        self.scaling_speed = 5
        self.pan_speed = 5

        # pca analysis stuff
        self.pca_data = {}
        self.pca_controls_built = False
        self.pca_lbox = None

        # COLOR LIST
        self.colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Pink', 'salmon1',
                       'snow4', 'gray98', 'black', 'DarkSlateGray3', 'DeepPink2', 'DarkGoldenrod2', 'DarkOrange1',
                       'tomato3', 'DarkSlateGray3', 'IndianRed4', 'antique white', 'LightSalmon4', 'gray18',
                       'LightBlue1', 'slate blue', 'gray78', 'LightSkyBlue3', 'thistle4', 'light slate blue']

        self.regression_mode = False  # this informs the program if the user is using the regression analysis tool or not.
        # This will be generalized in later versions

        # self.distribution = (
        #     "Uniform", "Uniform")  # used to keep track of the probability distribution the user wishes to use

        self.buildAxes()

    def buildMenus(self):

        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu=menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        menulist.append(filemenu)

        # create a tools menu
        toolsmenu = tk.Menu(menu)
        menu.add_cascade(label="Settings", menu=toolsmenu)
        menulist.append(toolsmenu)

        analysismenu = tk.Menu(menu)
        menu.add_cascade(label="Analysis", menu=analysismenu)
        menulist.append(analysismenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [
            ['Open \xE2\x8C\x98-O', 'Save-Image \xE2\x8C\x98-S ', 'Clear \xE2\x8C\x98-N', 'Quit  \xE2\x8C\x98-Q'],
            ['Set Transformation Parameters', 'Select Plot Features'],
            ['Linear Regression', 'Principal Component Analysis', 'Clustering Analysis', 'Classification']]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [[self.handleOpen, self.saveRegressionImage, self.reset, self.handleQuit],
                   [self.setTransformationParameters, self.handlePlotData, None],
                   [self.handleLinearRegression, self.handlePCA, self.handleClustering, self.handleClassifier]]

        # build the menu elements and callbacks
        for i in range(len(menulist)):
            for j in range(len(menutext[i])):
                if menutext[i][j] != '-':
                    menulist[i].add_command(label=menutext[i][j], command=menucmd[i][j])
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas(self.root, width=self.initDx, height=self.initDy)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        return

    # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        self.rightcntlframe = tk.Frame(self.root)
        self.rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame(self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN)
        sep.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label(self.rightcntlframe, text="Control Panel", width=20)
        label.pack(side=tk.TOP, pady=10)

        # # make a menubutton
        # self.colorOption = tk.StringVar(self.root)
        # self.colorOption.set("black")
        # colorMenu = tk.OptionMenu(rightcntlframe, self.colorOption,
        #                           "black", "blue", "red", "green")  # can add a command to the menu
        # colorMenu.pack(side=tk.TOP)
        #
        # # make a button in the frame
        # # and tell it to call the handleButton method when it is pressed.
        # button = tk.Button(rightcntlframe, text="Update Color",
        #                    command=self.handleButton1)
        # button.pack(side=tk.TOP)  # default side is top
        #
        # # make a button in the frame to generate random data points on the canvas
        # button2 = tk.Button(rightcntlframe, text="Generate Random", command=self.generateRandomData)
        # button2.pack(side=tk.TOP)
        #
        # # make a button in the frame to let the user choose what sort of probability distribution to use
        # button3 = tk.Button(rightcntlframe, text="Choose Distributions", command=self.chooseDist)
        # button3.pack(side=tk.TOP)
        #
        # # make a widget to allow users to specify the number of random points to be generated.
        # label3 = tk.Label(rightcntlframe, text="Number of Data Points", width=20)
        # label3.pack(side=tk.BOTTOM, pady=10)
        #
        # entry = tk.Entry(rightcntlframe, textvariable=self.num_pts)
        # entry.pack(side=tk.BOTTOM)

        return

    # Lets users open the csv files which they want to analyze.
    def handleOpen(self, event=None):
        fn = tkFileDialog.askopenfilename(parent=self.root, title="Choose a Data file", initialdir='.')
        if fn.split('.')[1] != "csv" and fn.split('.')[1] != "xls":
            tkMessageBox.showwarning("Open File", "Cannot open this file\n(%s)" % fn)
            return
        self.data = dt.Data(filename=fn)
        self.handlePlotData()
        self.filename = fn

    # allows users to choose which features from the data are to be displayed
    def handlePlotData(self):

        print "Getting headers!"
        if self.data is not None:
            d = ChooseAxesDialog(self.root, self.data.get_headers())

        else:
            tkMessageBox.showerror("No Open File", "Please open a file first!")
            print "First Open a Data File!!"
            return
        if d.result is not None:
            self.x_label_name.set(d.result[0][0])
            self.y_label_name.set(d.result[0][1])
            self.z_label_name.set(d.result[0][2])
            self.reset()
            spatial_headers = [h for h in d.result[0] if h != "None"]
            other_headers = d.result[1:]
            self.buildPoints(spatial_headers, other_headers, self.data)

    def buildPoints(self, spatial_headers, other_headers, data_obj):
        # clear all canvas data objects
        self.objects = []

        # normalize the required data
        # analysis.normalize_columns_separately(self.data, headers)
        # print "Self.data from buildpoints"
        # try:
        #     print self.data
        # except TypeError:
        #     pass
        # print "\n\n\n"

        # get all the columns by headers
        self.spatial_data = analysis.normalize_columns_separately(data_obj, spatial_headers)
        print self.spatial_data.T.tolist()
        if other_headers[0] != "None" and other_headers[2] != 1:
            self.color_data = analysis.normalize_columns_separately(data_obj, [other_headers[0]]).transpose().tolist()[
                0]
            print "HERE****"
            print self.color_data
        if other_headers[0] != "None" and other_headers[2] == 1:
            self.color_data = data_obj.get_data([other_headers[0]]).transpose().tolist()[0]
        if other_headers[1] != "None":
            self.size_data = analysis.normalize_columns_separately(data_obj, [other_headers[1]]).transpose().tolist()[
                0]
        print "Plot data from build points"
        # print self.spatial_data
        print "\n\n\n"

        if self.spatial_data.shape[1] == 2:  # if there are two columns to plot
            x_col = self.spatial_data.T.tolist()[0]
            y_col = self.spatial_data.T.tolist()[1]
            z_col = [0] * len(x_col)  # zeros in the z coordinate
            self.x_label_name.set(spatial_headers[0])
            self.y_label_name.set(spatial_headers[1])
            homg_col = [1] * len(x_col)  # homogeneous coordinate 1
            self.spatial_data = np.matrix([x_col, y_col, z_col, homg_col]).transpose()
        elif self.spatial_data.shape[1] == 3:  # if there are three columns to plot
            print "Three columns"
            x_col = self.spatial_data.T.tolist()[0]
            y_col = self.spatial_data.T.tolist()[1]
            z_col = self.spatial_data.T.tolist()[2]
            self.x_label_name.set(spatial_headers[0])
            self.y_label_name.set(spatial_headers[1])
            self.z_label_name.set(spatial_headers[2])
            homg_col = [1] * len(x_col)  # homogeneous coordinate 1
            self.spatial_data = np.matrix([x_col, y_col, z_col, homg_col]).transpose()
        # elif self.spatial_data.shape[1] == 4:  # if there's a color coordinate.
        #     x_col = self.spatial_data.T.tolist()[0]
        #     y_col = self.spatial_data.T.tolist()[1]
        #     z_col = self.spatial_data.T.tolist()[2]
        #     self.color_data = self.spatial_data.T.tolist()[3]
        #     self.x_label_name.set(spatial_headers[0])
        #     self.y_label_name.set(spatial_headers[1])
        #     self.z_label_name.set(spatial_headers[2])
        #     homg_col = [1] * len(x_col)  # homogeneous coordinate 1
        #     self.spatial_data = np.matrix([x_col, y_col, z_col, homg_col]).transpose()
        # elif self.spatial_data.shape[1] == 5:
        #     x_col = self.spatial_data.T.tolist()[0]
        #     y_col = self.spatial_data.T.tolist()[1]
        #     z_col = self.spatial_data.T.tolist()[2]
        #     self.color_data = self.spatial_data.T.tolist()[3]
        #     self.size_data = self.spatial_data.T.tolist()[4]
        #     self.x_label_name.set(spatial_headers[0])
        #     self.y_label_name.set(spatial_headers[1])
        #     self.z_label_name.set(spatial_headers[2])
        #     homg_col = [1] * len(x_col)  # homogeneous coordinate 1
        #     self.spatial_data = np.matrix([x_col, y_col, z_col, homg_col]).transpose()
        elif self.spatial_data.shape[1] == 1:  # histogram
            hist = np.histogram(self.spatial_data.T.tolist()[0], 10)
            maxHist = hist[0].max(0)
            for i in range(10):
                for j in range(hist[0][i]):
                    point = np.matrix([[float(i) / 10, float(j) / maxHist, 0, 1]])
                    if i == 0 and j == 0:
                        temp_data = point
                    else:
                        temp_data = np.vstack((temp_data, point))
            self.spatial_data = temp_data
            self.x_label_name.set("Count")

        print self.spatial_data

        # build the view transformation matrix
        vtm = self.view.build()

        print vtm

        # create the transformed plot data matrix
        pts = (vtm * self.spatial_data.T).T

        for i in range(self.spatial_data.shape[0]):
            # print "Drawing"
            x = pts[i, 0]
            y = pts[i, 1]
            dx = 3
            if self.color_data is None and self.size_data is None:
                self.objects.append(self.canvas.create_oval(x - dx, y - dx, x + dx, y + dx, fill='black', outline=''))
            elif self.color_data is not None and self.size_data is None:
                if other_headers[2] != 1:
                    rgb = "#%02x%02x%02x" % (
                        255 - int(self.color_data[i] * 255), 255 - int(self.color_data[i] * 255),
                        int(self.color_data[i] * 255))
                    self.objects.append(self.canvas.create_oval(x - dx, y - dx, x + dx, y + dx, fill=rgb, outline=''))
                else:
                    self.objects.append(self.canvas.create_oval(x - dx, y - dx, x + dx, y + dx,
                                                                fill=self.colors[int(self.color_data[i])], outline=''))

            else:
                if other_headers[2] != 1:
                    rgb = "#%02x%02x%02x" % (
                        255 - int(self.color_data[i] * 255), 255 - int(self.color_data[i] * 255),
                        int(self.color_data[i] * 255))
                    self.objects.append(
                        self.canvas.create_oval(x - dx * self.size_data[i] - 1, y - dx * self.size_data[i] - 1,
                                                x + dx * self.size_data[i] + 1,
                                                y + dx * self.size_data[i] + 1, fill=rgb, outline=''))
                else:
                    self.objects.append(self.canvas.create_oval(x - dx, y - dx, x + dx, y + dx,
                                                                fill=self.colors[int(self.color_data[i])], outline=''))

        # make the legend
        legend_text = """
-------------------
   Color     %s
   Size     %s
-------------------"""

        legend_frame = tk.LabelFrame(self.canvas, text='Legend', padx=5, pady=5)
        legend_label = tk.Label(legend_frame, text=legend_text % (other_headers[0], other_headers[1]))
        legend_label.pack()

        self.canvas.create_window(900, 600, window=legend_frame, anchor=tk.E)

    # updates the location of the data points when the view reference parameters are changed
    def updatePoints(self):
        # create the view transformation matrix
        vtm = self.view.build()

        # create the transformed plot data matrix
        pts = (vtm * self.spatial_data.T).T

        # update locations of data points
        for i in range(len(self.objects)):
            x = pts[i, 0]
            y = pts[i, 1]
            dx = 3
            if self.size_data is not None:

                self.canvas.coords(self.objects[i], x - dx * self.size_data[i] - 1, y - dx * self.size_data[i] - 1,
                                   x + dx * self.size_data[i] + 1, y + dx * self.size_data[i] + 1)
            else:
                self.canvas.coords(self.objects[i], x - dx, y - dx, x + dx, y + dx)

    # build the axis lines and draw them on the canvas
    def buildAxes(self):
        vtm = self.view.build()
        pts = (vtm * self.axes.T).T
        # print pts
        x_axis = self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1])
        y_axis = self.canvas.create_line(pts[2, 0], pts[2, 1], pts[3, 0], pts[3, 1])
        z_axis = self.canvas.create_line(pts[4, 0], pts[4, 1], pts[5, 0], pts[5, 1])
        x_label = self.canvas.create_text(pts[1, 0], pts[1, 1], text=self.x_label_name.get())
        y_label = self.canvas.create_text(pts[3, 0], pts[3, 1], text=self.y_label_name.get())
        z_label = self.canvas.create_text(pts[5, 0], pts[5, 1], text=self.z_label_name.get())
        self.axis_labels = [x_label, y_label, z_label]
        self.axis_lines = [x_axis, y_axis, z_axis]

    # update the axis endpoints when the user provides input
    def updateAxes(self):
        vtm = self.view.build()
        pts = (vtm * self.axes.T).T
        for i in range(len(self.axis_lines)):
            self.canvas.coords(self.axis_lines[i], pts[2 * i, 0], pts[2 * i, 1], pts[2 * i + 1, 0], pts[2 * i + 1, 1])
            # self.canvas.coords(self.axis_labels[i], pts[2 * i + 1, 0], pts[2 * i + 1, 1])
        self.canvas.delete(self.axis_labels[0], self.axis_labels[1], self.axis_labels[2])
        x_label = self.canvas.create_text(pts[1, 0], pts[1, 1], text=self.x_label_name.get())
        y_label = self.canvas.create_text(pts[3, 0], pts[3, 1], text=self.y_label_name.get())
        z_label = self.canvas.create_text(pts[5, 0], pts[5, 1], text=self.z_label_name.get())
        self.axis_labels = [x_label, y_label, z_label]

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind('<Button-1>', self.handleMouseButton1)
        self.canvas.bind('<Control-Button-1>', self.handleMouseButton2)
        self.canvas.bind('<Button-2>', self.handleMouseButton2)
        self.canvas.bind('<B1-Motion>', self.handlePanning)
        self.canvas.bind('<B2-Motion>', self.handleScaling)
        self.canvas.bind('<Control-B1-Motion>', self.handleScaling)
        self.canvas.bind('<Button-3>', self.handleButton3Click)
        self.canvas.bind('<Shift-Command-Button-1>', self.deleteDataPoint)
        self.canvas.bind('<B3-Motion>', self.handleRotation)

        # bind command sequences to the root window
        self.root.bind('<Command-q>', self.handleQuit)
        self.root.bind('<Command-n>', self.reset)
        self.root.bind('<Command-o>', self.handleOpen)
        self.root.bind('<Command-s>', self.saveRegressionImage)

    def handleQuit(self, event=None):
        print 'Terminating'
        self.root.destroy()

    def handleButton1(self):
        print 'handling command button:', self.colorOption.get()
        for obj in self.objects:
            self.canvas.itemconfig(obj, fill=self.colorOption.get())

    def setTransformationParameters(self):
        print 'handling Data Transformation stuff'
        dialog = TPDialog(self.root, "Choose Transformation Parameters", self.scaling_speed, self.pan_speed,
                          self.rotation_speed)
        if dialog.result is not None:
            self.scaling_speed = max(1, min(dialog.result[0], 10))
            self.pan_speed = max(1, min(dialog.result[1], 10))
            self.rotation_speed = max(1, min(dialog.result[2], 10))

    def handleMouseButton1(self, event):
        print 'handle mouse button 1: %d %d' % (event.x, event.y)
        self.baseClick = (event.x, event.y)

    def handleMouseButton2(self, event):
        self.baseClick = (event.x, event.y)
        self.baseExtent = self.view.clone().extent

        print 'handle mouse button 2: %d %d' % (event.x, event.y)

    # This is called if the first mouse button is being moved.
    # Causes the axes to pan
    def handlePanning(self, event):
        # old code. make it work later.
        # diff = (event.x - self.baseClick[0], event.y - self.baseClick[1])
        # for obj in self.objects:
        #     loc = self.canvas.coords(obj)
        #     self.canvas.coords(obj, loc[0] + diff[0], loc[1] + diff[1], loc[2] + diff[0], loc[3] + diff[1])
        #
        # # update base click
        # self.baseClick = (event.x, event.y)
        # print 'handle button1 motion %d %d' % (diff[0], diff[1])

        # calculate the differential motion
        delta1 = float(event.x - self.baseClick[0])
        delta1 /= self.canvas.winfo_width()
        delta1 *= self.view.extent[0]
        delta1 *= self.pan_speed

        delta2 = float(event.y - self.baseClick[1])
        delta2 /= self.canvas.winfo_height()
        delta2 *= self.view.extent[1]
        delta2 *= self.pan_speed

        # apply the differential change to the vrp
        self.view.vrp = self.view.vrp + delta1 * self.view.u + delta2 * self.view.vup
        self.updateAxes()
        self.updateFits()
        if len(self.objects) > 0:
            self.updatePoints()

        # reset baseClick
        self.baseClick = (event.x, event.y)

    # This is called if the second button of a real mouse has been pressed
    # and the mouse is moving. Or if the control key is held down while
    # a person moves their finger on the track pad.
    def handleScaling(self, event):
        print 'handle button 2 motion'
        # SCALING FOR DATA POINTS
        #    loc = self.canvas.coords(obj)
        #    #scale = 1 - 1/200*(event.y - self.baseClick[1])
        #    if event.y < self.baseClick[1]:
        #        self.canvas.coords(obj, loc[0] - 0.2, loc[1] - 0.2, loc[2] + 0.2, loc[3] + 0.2)
        #        self.size += 0.005
        #    else:
        #        self.canvas.coords(obj, loc[0] + 0.2, loc[1] + 0.2, loc[2] - 0.2, loc[3] - 0.2)
        #        self.size -= 0.005
        # #self.canvas.coords(obj,loc[0]*scale*self.size/10,loc[1]*scale*self.size/10, loc[2]*scale*self.size/10, loc[3]*scale*self.size/10)

        # scaling for axes
        dy = float(event.y - self.baseClick[1])
        scale_factor = 1 + self.scaling_speed * dy / 1000000 * self.canvas.winfo_height()
        bounded_scale_factor = max(0.1, min(scale_factor, 3.0))
        print bounded_scale_factor
        self.view.extent = [bounded_scale_factor * self.baseExtent[0], bounded_scale_factor * self.baseExtent[1],
                            bounded_scale_factor * self.baseExtent[2]]
        self.updateAxes()
        self.updateFits()
        if len(self.objects) > 0:
            self.updatePoints()

    # handle right click by creating a data point at the location.
    def handleButton3Click(self, event):
        # set up for rotation
        print "Handing mouse button 3 at %d,%d" % (event.x, event.y)
        self.baseClick2 = (event.x, event.y)
        self.viewClone = self.view.clone()

    def handleRotation(self, event=None):
        delta0 = float(event.x - self.baseClick2[0]) / (0.5 * self.canvas.winfo_width()) * math.pi
        delta1 = float(event.y - self.baseClick2[1]) / (0.5 * self.canvas.winfo_height()) * math.pi

        self.view = self.viewClone.clone()

        self.view.rotateVRC(delta0, -delta1)
        self.updateAxes()
        self.updateFits()
        if len(self.objects) > 0:
            self.updatePoints()

    # handle the 'Generate Random' button
    def generateRandomData(self, event=None):
        print "X-Direction:", self.distribution[0]
        print "Y-Direction:", self.distribution[1]
        dx = 3
        dy = 3
        for i in range(int(self.num_pts.get())):
            if self.distribution[0] == "Uniform":
                x = random.randint(dx, self.canvas.winfo_width() - dx)
            else:
                # According to Adam Carlson, there's 99.8% chance that data in a normal distribution is within 3 standard deviations of the mean.
                # mu = mean, sigma = standard deviation. Look and infer.
                x = random.gauss(mu=(self.canvas.winfo_width() - dx) / 2, sigma=self.canvas.winfo_width() / 6)
            if self.distribution[1] == "Uniform":
                y = random.randint(dy, self.canvas.winfo_height() - dy)
            else:
                y = random.gauss(mu=(self.canvas.winfo_height() - dy) / 2, sigma=self.canvas.winfo_height() / 6)

            # make sure that the coords are not out of bounds!
            x %= self.canvas.winfo_width() - dx / 2
            y %= self.canvas.winfo_height() - dy / 2

            pt = self.canvas.create_oval(x - dx, y - dy, x + dx, y + dy, fill=self.colorOption.get(), outline='')
            self.objects.append(pt)

    # handle the clear command
    def clearData(self, event=None):
        self.objects = []
        self.canvas.delete("all")
        self.size = 10

    # reset the view volume axes
    def reset(self, event=None):
        self.clearData()
        self.view.reset()
        self.color_data = None
        self.size_data = None
        self.linreg_endpoints = None
        self.linreg_line_objects = []
        self.regression_mode = False
        self.buildAxes()

    # handle the choose distribution button
    def chooseDist(self, event=None):
        dialog = DistDialog(self.root, "Choose Distributions")
        if dialog.result[0] is not None and dialog.result[1] is not None:
            self.distribution = dialog.result

    # handle the delete data point button
    def deleteDataPoint(self, event=None):
        print "deleting"
        self.baseClick = (event.x, event.y)
        for obj in self.objects:
            loc1 = self.canvas.coords(obj)
            real_loc = [(loc1[0] + loc1[2]) / 2, (loc1[1] + loc1[3]) / 2]
            if get_distance(real_loc, self.baseClick) < self.size:
                print real_loc
                print self.baseClick
                print get_distance(real_loc, self.baseClick)
                self.canvas.delete(obj)
                self.objects.remove(obj)
                break

    # handle linear regression command
    def handleLinearRegression(self):
        if self.data is None:
            tkMessageBox.showerror("No Open File", "Please open a file first!")
            return
        d = LinearRegressionDialog(self.root, self.data.get_headers())
        if d.result is not None:
            if d.result[0] == "None" or d.result[1] == "None":
                tkMessageBox.showwarning(title="Incorrect Parameters", message="Option cannot be 'None'")
            else:
                self.objects = []
                self.linreg_endpoints = None
                self.linreg_line_objects = []
                self.reset()
                self.updateAxes()
                self.regression_mode = True
                self.buildLinearRegression(headers=d.result)

    # build and display the regression on the plot
    def buildLinearRegression(self, headers):
        # raw data
        x_raw = self.data.get_data([headers[0]]).T.tolist()[0]
        y_raw = self.data.get_data([headers[1]]).T.tolist()[0]

        self.x_label_name.set(value=headers[0])
        self.y_label_name.set(value=headers[1])

        self.updateAxes()

        # normalized
        x_col = analysis.normalize_columns_separately(self.data, [headers[0]]).T.tolist()[0]
        y_col = analysis.normalize_columns_separately(self.data, [headers[1]]).T.tolist()[0]
        z_col = [0] * len(x_col)
        homg = [1] * len(x_col)

        # build raw data matrix
        self.spatial_data = np.matrix([x_col, y_col, z_col, homg]).T
        print self.spatial_data

        # build vtm
        vtm = self.view.build()

        # build adjusted
        pts = (vtm * self.spatial_data.T).T

        for i in range(self.spatial_data.shape[0]):
            # print "Drawing"
            x = pts[i, 0]
            y = pts[i, 1]
            dx = 3
            if self.color_data is None and self.size_data is None:
                self.objects.append(self.canvas.create_oval(x - dx, y - dx, x + dx, y + dx, fill='black', outline=''))

        # regression information
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_raw, y_raw)

        # normalized endpoints for best fit line
        y0 = (min(x_raw) * slope + intercept - min(y_raw)) / (max(y_raw) - min(y_raw))
        y1 = (max(x_raw) * slope + intercept - min(y_raw)) / (max(y_raw) - min(y_raw))

        self.linreg_endpoints = np.matrix([[0.0, y0, 0.0, 1.0], [1.0, y1, 0.0, 1.0]])
        pts = (vtm * self.linreg_endpoints.T).T

        print pts

        self.linreg_line_objects.append(self.canvas.create_line(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], fill="red"))

        self.canvas.create_text(800, 50, text="Slope: %f" % slope)
        self.canvas.create_text(800, 70, text="R Value: %f" % r_value)
        self.canvas.create_text(800, 90, text="Y-Intercept: %f" % intercept)
        self.canvas.create_text(800, 110, text="P value: %f" % p_value)

        # generate an analysis file.
        analysis.test_lin_reg(self.filename, [headers[0]], headers[1])

    # update the position of the regression line
    def updateFits(self):
        if self.linreg_endpoints is None:
            return

        vtm = self.view.build()
        pts = (vtm * self.linreg_endpoints.T).T
        for item in self.linreg_line_objects:
            self.canvas.coords(item, pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1])

    # saves an image of the regression analysis using matplotlib
    def saveRegressionImage(self, event=None):
        if not self.regression_mode:
            tkMessageBox.showerror("Can't Save Image", "This feature only works for regression analysis currently.")
            return
        else:
            x = analysis.normalize_columns_separately(self.data, [self.x_label_name.get()]).T.tolist()[0]
            y = analysis.normalize_columns_separately(self.data, [self.y_label_name.get()]).T.tolist()[0]

            plt.plot(x, y, 'o')
            plt.plot([min(x), max(x)], [self.linreg_endpoints[0, 1], self.linreg_endpoints[1, 1]])
            plt.ylabel(self.y_label_name.get())
            plt.xlabel(self.x_label_name.get())

            plt.show()

    ### BEGIN PCA STUFF ###

    def handlePCA(self, event=None):
        if self.data is None:
            tkMessageBox.showerror("No File Open", "Please open a file first!")
            return
        d = PCADialog(self.root, self.data.get_headers())
        if d.result is None:
            return
        if d.result[2] in self.pca_data.keys():
            tkMessageBox.showwarning("Replacing Saved Analysis",
                                     "Please delete old analysis before creating another with the same name.")
            return
        if d.result[0] == 1:
            self.pca_data[d.result[2]] = analysis.pca(self.data, d.result[1])
            self.pca_data[d.result[2]].write_to_file(d.result[2], self.pca_data[d.result[2]].get_headers())
            print "Normalizing: True"
        else:
            self.pca_data[d.result[2]] = analysis.pca(self.data, d.result[1], False)
            self.pca_data[d.result[2]].write_to_file(d.result[2], self.pca_data[d.result[2]].get_headers())
            print "Normalizing: False"
        if not self.pca_controls_built:
            self.buildPCAControls()

        self.pca_lbox.insert(tk.END, d.result[2])

    def handle_pca_button(self, event=None):
        name = self.pca_lbox.get(self.pca_lbox.curselection()[0])
        dataobj = self.pca_data[name]
        d = ChooseAxesDialog(self.root, dataobj.get_headers())
        if d.result is not None:
            self.x_label_name.set(d.result[0][0])
            self.y_label_name.set(d.result[0][1])
            self.z_label_name.set(d.result[0][2])
            self.reset()
            spatial_headers = [h for h in d.result[0] if h != "None"]
            other_headers = d.result[1:]
            print "\n\n Headers for point building"
            print spatial_headers, other_headers
            self.buildPoints(spatial_headers, other_headers, dataobj)

    def handle_delete_button(self):
        if len(self.pca_lbox.curselection()) > 0:
            self.pca_data.pop(self.pca_lbox.get(self.pca_lbox.curselection()[0]))
            self.pca_lbox.delete(self.pca_lbox.curselection()[0])

    def show_eigen_table(self):
        name = self.pca_lbox.get(self.pca_lbox.curselection()[0])
        dataobj = self.pca_data[name]
        d = EigenTable(self.root, dataobj)

    def buildPCAControls(self):
        label1 = tk.Label(self.rightcntlframe, text="PCA Analysis")
        label1.pack(side=tk.TOP)
        self.pca_lbox = tk.Listbox(self.rightcntlframe, selectmode=tk.SINGLE)
        self.pca_lbox.pack(side=tk.TOP)
        self.pca_controls_built = True
        plot_button = tk.Button(self.rightcntlframe, text="Plot PCA", command=self.handle_pca_button)
        plot_button.pack(side=tk.TOP)
        delete_button = tk.Button(self.rightcntlframe, text="Delete Analysis", command=self.handle_delete_button)
        delete_button.pack(side=tk.TOP)
        table_button = tk.Button(self.rightcntlframe, text="Show Eigenvectors", command=self.show_eigen_table)
        table_button.pack(side=tk.TOP)

    ### END PCA STUFF ###

    # clustering stuff
    def handleClustering(self):
        if self.data is None:
            tkMessageBox.showerror("No File Open", "Please open a file first!")
            return
        d = ClusteringDialog(self.root, self.data.get_headers())
        headers = d.result[0]
        metric = d.result[1]
        K = d.result[2]
        means, codebook, error = analysis.kmeans(self.data, headers, K, metric)
        print codebook.T.tolist()
        self.data.add_column("cluster", "numeric", codebook.T.tolist()[0])

    def handleClassifier(self):
        # if self.data is None:
        #     tkMessageBox.showerror("No File Open", "Please open a file first!")
        #     return
        d = ClassifierDialog(self.root)
        if d.result is None:
            return
        if d.result[3] is None:
            cm.main(["asd", d.result[0], d.result[1], d.result[2]])
        else:
            cm.main(["asd", d.result[0], d.result[1], d.result[2], d.result[3], d.result[4]])

    def main(self):
        print 'Entering main loop'
        self.root.mainloop()


class Dialog(tk.Toplevel):
    def __init__(self, parent, title=None, default_scale=None, default_pan=None, default_rotate=None):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        # set the default transformation parameters for use by subclass
        self.scale = default_scale
        self.pan = default_pan
        self.rotate = default_rotate

        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
                                  parent.winfo_rooty() + 50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set()  # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()

    def cancel(self, event=None):

        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1  # override

    def apply(self):

        pass  # override


# Dialog for setting the random distribution
class DistDialog(Dialog):
    # Sets up the GUI for the dialog with a Listbox containing the options
    def body(self, master):
        # label the listboxes
        tk.Label(master, text="X-Distribution: ").grid(row=0)
        tk.Label(master, text="Y-Distribution: ").grid(row=1)

        # set up the listboxes in the mode defined in the project
        self.xbox = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
        self.ybox = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)

        # set up the grid layout
        self.xbox.grid(row=0, column=1)
        self.ybox.grid(row=1, column=1)

        # add items
        self.xbox.insert(tk.END, "Gaussian")
        self.xbox.insert(tk.END, "Uniform")

        self.ybox.insert(tk.END, "Gaussian")
        self.ybox.insert(tk.END, "Uniform")

        # highlight the currently selected choices (extension)
        # if distribution[0] == "Uniform":
        #     self.xbox.selection_set(1)
        # else:
        #     self.xbox.select_set(0)
        #
        # if distribution[1] == "Uniform":
        #     self.ybox.selection_set(1)
        # else:
        #     self.ybox.selection_set(0)
        return self.xbox

    def apply(self):
        self.result = (self.xbox.get(tk.ACTIVE), self.ybox.get(tk.ACTIVE))


# the dialog which helps users set the transformation parameters
# for scaling, panning and rotating the axes.
class TPDialog(Dialog):
    def body(self, master):
        # set up description
        label = tk.Label(master, text="1-Slow, 10-Fast. Choose wisely.").grid(row=0)

        # set up labels for the respective entries
        tk.Label(master, text="Scale-Speed").grid(row=1)
        tk.Label(master, text="Pan-Speed").grid(row=2)
        tk.Label(master, text="Rotate-Speed").grid(row=3)

        # set up entry boxes with default values
        self.scale_speed_entry = tk.Entry(master)
        self.scale_speed_entry.grid(row=1, column=1)
        self.scale_speed_entry.insert(0, str(self.scale))

        self.pan_speed_entry = tk.Entry(master)
        self.pan_speed_entry.grid(row=2, column=1)
        self.pan_speed_entry.insert(0, str(self.pan))

        self.rotate_speed_entry = tk.Entry(master)
        self.rotate_speed_entry.grid(row=3, column=1)
        self.rotate_speed_entry.insert(0, str(self.rotate))

        return self.scale_speed_entry

    def apply(self):
        try:
            self.result = (
                int(self.scale_speed_entry.get()), int(self.pan_speed_entry.get()), int(self.rotate_speed_entry.get()))
        except ValueError:
            self.result = (self.scale, self.pan, self.rotate)


# Dialog class which allows users to choose which data features to display
class ChooseAxesDialog(Dialog):
    def __init__(self, parent, headers):
        self.headers = headers
        self.headers.append("None")
        self.x = tk.StringVar(master=parent, value=headers[0])
        self.y = tk.StringVar(master=parent, value=headers[0])
        self.z = tk.StringVar(master=parent, value=headers[0])
        self.color = tk.StringVar(master=parent, value='None')
        self.size = tk.StringVar(master=parent, value='None')
        self.use_preselected_color = tk.IntVar(parent, 0)
        Dialog.__init__(self, parent, title="Choose Axes")

    def body(self, master):
        # set up description
        tk.Label(master, text='Choose at least one spatial direction.').grid(row=0)
        tk.Label(master, text='Do not leave "None" between two choices').grid(row=1)

        # set up labels for the respective entries
        tk.Label(master, text="X-Coordinate").grid(row=2)
        tk.Label(master, text="Y-Coordinate").grid(row=3)
        tk.Label(master, text="Z-Coordinate").grid(row=4)
        tk.Label(master, text="Color").grid(row=5)
        tk.Label(master, text="Size").grid(row=6)

        # set up the option menus
        x_choice = apply(tk.OptionMenu, (master, self.x) + tuple(self.headers))
        x_choice.grid(row=2, column=1)

        y_choice = apply(tk.OptionMenu, (master, self.y) + tuple(self.headers))
        y_choice.grid(row=3, column=1)

        z_choice = apply(tk.OptionMenu, (master, self.z) + tuple(self.headers))
        z_choice.grid(row=4, column=1)

        color_choice = apply(tk.OptionMenu, (master, self.color) + tuple(self.headers))
        color_choice.grid(row=5, column=1)

        size_choice = apply(tk.OptionMenu, (master, self.size) + tuple(self.headers))
        size_choice.grid(row=6, column=1)

        self.checkbutton = tk.Checkbutton(master, text="Use Preselected Colors",
                                          variable=self.use_preselected_color)

        self.checkbutton.grid(row=7)

        self.checkbutton.config(state="disabled")

        # self.use_preselected_color.trace('w',self.check_function)
        self.color.trace('w', self.check_function)

    def check_function(self, *args):
        if self.color.get() == "cluster" or self.color.get() == "class":
            self.checkbutton.config(state="active")
            print "enabled"

    def apply(self):
        try:
            self.result = ((self.x.get(), self.y.get(), self.z.get()), self.color.get(), self.size.get(),
                           self.use_preselected_color.get())
        except ValueError:
            print "Value Error"


# Dialog box to handle users linear regression variable choices
class LinearRegressionDialog(Dialog):
    def __init__(self, parent, headers):
        self.x = tk.StringVar(master=parent, value="None")
        self.y = tk.StringVar(master=parent, value="None")
        self.headers = headers
        Dialog.__init__(self, parent, title="Choose Regression Parameters")

    def body(self, master):
        tk.Label(master, text='Choose variables for regression').grid(row=0)
        tk.Label(master, text='Independent variable (X)').grid(row=1)
        tk.Label(master, text='Dependent variable (Y)').grid(row=2)

        x_choice = apply(tk.OptionMenu, (master, self.x) + tuple(self.headers))
        x_choice.grid(row=1, column=1)

        y_choice = apply(tk.OptionMenu, (master, self.y) + tuple(self.headers))
        y_choice.grid(row=2, column=1)

    def apply(self):
        self.result = (self.x.get(), self.y.get())


# Dialog box to handle PCA analysis
class PCADialog(Dialog):
    def __init__(self, parent, headers):
        self.headers = headers
        self.normalize = tk.IntVar(parent, value=0)
        self.analysis_name = tk.StringVar(parent, value="***Replace with name***")
        print self.normalize, self.analysis_name
        self.header_box = None
        Dialog.__init__(self, parent, "Principal Component Analysis")

    def body(self, master):
        tk.Label(master, text="Choose at least three columns for analysis").grid(row=0)
        self.header_box = tk.Listbox(master, selectmode=tk.MULTIPLE)
        self.header_box.grid(row=4)
        for item in self.headers:
            self.header_box.insert(tk.END, item)
        tk.Checkbutton(master, text="Normalize", variable=self.normalize).grid(row=3)
        tk.Label(master, text="Analysis name").grid(row=1)
        tk.Entry(master, textvariable=self.analysis_name).grid(row=2)

    def apply(self):
        columns = []

        for i in self.header_box.curselection():
            columns.append(self.header_box.get(i))

        if len(columns) < 3:
            tkMessageBox.showerror("Not enough headers", "Please choose at least three headers")
            return

        self.result = (self.normalize.get(), columns, self.analysis_name.get())


# Dialog box to show eigne data for PCA
class EigenTable(Dialog):
    def __init__(self, parent, dataobj):
        self.headers = dataobj.get_headers()
        self.orig_headers = dataobj.get_data_headers()
        self.evals = dataobj.get_eigenvalues().tolist()
        self.evecs = dataobj.get_eigenvectors().tolist()
        self.pdata = dataobj.get_matrix_data().T.tolist()
        Dialog.__init__(self, parent, "Principal Component Analysis")

    def body(self, master):
        tk.Label(master, text="E-vec").grid(row=0, column=0)
        tk.Label(master, text="E-val").grid(row=0, column=1)

        for i in range(len(self.orig_headers)):
            tk.Label(master, text=self.orig_headers[i]).grid(row=0, column=i + 2)

        for j in range(len(self.headers)):
            tk.Label(master, text=self.headers[j]).grid(row=j + 1, column=0)

        for k in range(len(self.evals[0])):
            number = "%.4f" % float(self.evals[0][k])
            tk.Label(master, text=number).grid(row=k + 1, column=1)

        for rows in range(len(self.evecs)):
            for columns in range(len(self.evecs[0])):
                number = "%.4f" % float(self.evecs[rows][columns])
                tk.Label(master, text=number).grid(row=rows + 1, column=columns + 2)

    def apply(self):
        pass


# Dialog box to handle PCA analysis
class ClusteringDialog(Dialog):
    def __init__(self, parent, headers):
        self.headers = headers
        self.metric = tk.StringVar(parent, "euclidean")
        self.clusters = tk.StringVar(parent, "2")
        self.header_box = None
        Dialog.__init__(self, parent, "Principal Component Analysis")

    def body(self, master):
        tk.Label(master, text="Choose features for clustering analysis").grid(row=0)
        self.header_box = tk.Listbox(master, selectmode=tk.MULTIPLE)
        self.header_box.grid(row=4)
        for item in self.headers:
            self.header_box.insert(tk.END, item)
        tk.Label(master, text="Algorithm").grid(row=5, column=0)
        tk.OptionMenu(master, self.metric, "euclidean", "cityblock", "cosine", "canberra", "braycurtis").grid(row=6,
                                                                                                              column=0)
        tk.Label(master, text="Clusters").grid(row=7)
        k_menu = apply(tk.OptionMenu, (master, self.clusters) + tuple(range(2, 26)))
        k_menu.grid(row=8)
        # tk.Label(master, text="Analysis name").grid(row=2)
        # tk.Entry(master, textvariable=self.analysis_name).grid(row=3)

    def apply(self):
        columns = []

        for i in self.header_box.curselection():
            columns.append(self.header_box.get(i))

        if len(columns) < 3:
            tkMessageBox.showerror("Not enough headers", "Please choose at least three headers")
            return

        self.result = (columns, self.metric.get(), int(self.clusters.get()))


class ClassifierDialog(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.training_data = None
        self.testing_data = None
        self.training_labels = None
        self.testing_labels = None
        self.labels_given = tk.IntVar(parent, 1)
        self.algorithm = tk.StringVar(parent, value="Naive Bayes")
        self.labels_given.trace('w', self.tracker_function)
        Dialog.__init__(self, parent, "Choose files for Classification")

    def body(self, master):
        tk.Checkbutton(master, text="Labels included", variable=self.labels_given).grid(row=0)
        self.menu = apply(tk.OptionMenu, (master, self.algorithm) + tuple(["Naive Bayes", "K-Nearest Neighbors"]))
        self.menu_label = tk.Label(master, text="Algorithm")
        self.menu_label.grid(row=1, column=0)
        self.menu.grid(row=1, column=1)
        self.training_data_button = tk.Button(master, text="Choose Training Data", command=self.handle_training_button)
        self.training_data_button.grid(row=2)
        self.testing_data_button = tk.Button(master, text="Choose Testing Data", command=self.handle_testing_button)
        self.testing_data_button.grid(row=3)
        self.training_label_button = tk.Button(master, text="Choose Training labels",
                                               command=self.handle_training_label_button)
        self.training_label_button.config(state="disabled")
        self.training_label_button.grid(row=4)
        self.testing_label_button = tk.Button(master, text="Choose Testing labels",
                                              command=self.handle_testing_label_button)
        self.testing_label_button.config(state="disabled")
        self.testing_label_button.grid(row=5)

    def apply(self):
        self.result = (
        self.algorithm.get(), self.training_data, self.testing_data, self.training_labels, self.testing_labels)

    def handle_training_button(self):
        self.training_data = tkFileDialog.askopenfilename(parent=self.parent, title="Choose a training data file")

    def handle_testing_button(self):
        self.testing_data = tkFileDialog.askopenfilename(parent=self.parent, title="Choose a testing data file")

    def handle_training_label_button(self):
        self.training_labels = tkFileDialog.askopenfilename(parent=self.parent,
                                                            title="Choose separate training labels file")

    def handle_testing_label_button(self):
        self.testing_labels = tkFileDialog.askopenfilename(parent=self.parent,
                                                           title="Choose separate testing labels file")

    def tracker_function(self, *args):
        if self.labels_given.get() == 1:
            self.training_label_button.config(state="disabled")
            self.testing_label_button.config(state="disabled")
        else:
            self.training_label_button.config(state="normal")
            self.testing_label_button.config(state="normal")


if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()
