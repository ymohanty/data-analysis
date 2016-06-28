__author__ = "Yashaswi Mohanty"
__email__ = "ymohanty@colby.edu"
__version__ = "2/26/16"

import numpy as np
import math
import copy


# static function which normalizes vectors
def normalize(vector):
    length = math.sqrt(vector[0, 0] * vector[0, 0] + vector[0, 1] * vector[0, 1] + vector[0, 2] * vector[0, 2])
    vector[0, 0] /= length
    vector[0, 1] /= length
    vector[0, 2] /= length


class View:
    # constructor
    def __init__(self):
        self.vrp = np.matrix([0.5, 0.5, 1.0])
        self.vpn = np.matrix([0.0, 0.0, -1.0])
        self.vup = np.matrix([0.0, 1.0, 0.0])
        self.u = np.matrix([-1.0, 0.0, 0.0])
        self.extent = [1.0, 1.0, 1.0]
        self.screen = [400.0, 400.0]
        self.offset = [20.0, 20.0]

    # reset the fields to default values
    def reset(self):
        self.vrp = np.matrix([0.5, 0.5, 1])
        self.vpn = np.matrix([0.0, 0.0, -1.0])
        self.vup = np.matrix([0.0, 1.0, 0.0])
        self.u = np.matrix([-1.0, 0.0, 0.0])
        self.extent = [1.0, 1.0, 1.0]
        self.screen = [400.0, 400.0]
        self.offset = [20.0, 20.0]

    # build a view matrix from the current viewing parameters
    def build(self):
        vtm = np.identity(4, float)

        # move vrp to the origin
        t1 = np.matrix([[1, 0, 0, -self.vrp[0, 0]],
                        [0, 1, 0, -self.vrp[0, 1]],
                        [0, 0, 1, -self.vrp[0, 2]],
                        [0, 0, 0, 1]])

        # right now vtm = t1
        vtm = t1 * vtm

        # calculate the view reference axes
        tu = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn.copy()

        # normalize view reference axes
        normalize(tu)
        normalize(tvpn)
        normalize(tvup)

        # copy new view reference axes to fields
        self.u = tu.copy()
        self.vpn = tvpn.copy()
        self.vup = tvup.copy()

        # align the axes
        r1 = np.matrix([[tu[0, 0], tu[0, 1], tu[0, 2], 0.0],
                        [tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0],
                        [tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        vtm = r1 * vtm

        # translate the lower left corner of the view space to the origin.
        vtm = np.matrix([[1, 0, 0, 0.5 * self.extent[0]],
                         [0, 1, 0, 0.5 * self.extent[1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) * vtm

        # use the extent and screen size to scale to the screen
        vtm = np.matrix([[-self.screen[0] / self.extent[0], 0, 0, 0],
                         [0, -self.screen[1] / self.extent[1], 0, 0],
                         [0, 0, 1 / self.extent[2], 0],
                         [0, 0, 0, 1]]) * vtm

        # translate the lower left corner to the origin and add the view offset, which gives a little buffer
        # around the top and left edges of the window.
        vtm = np.matrix([[1, 0, 0, self.screen[0] + self.offset[0]],
                         [0, 1, 0, self.screen[1] + self.offset[1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]) * vtm
        return vtm

    # returns a deep copy of the current view obejct
    def clone(self):
        new_view = View()
        new_view.vrp = self.vrp.copy()
        new_view.vpn = self.vpn.copy()
        new_view.vup = self.vup.copy()
        new_view.u = self.u.copy()
        new_view.extent = copy.deepcopy(self.extent)
        new_view.screen = copy.deepcopy(self.screen)
        new_view.offset = copy.deepcopy(self.offset)
        return new_view

    def rotateVRC(self, a1, a2):
        t1 = np.matrix([[1, 0, 0, -1 * (self.vrp[0, 0] + self.vpn[0, 0] * self.extent[2] * 0.5)],
                        [0, 1, 0, -1 * (self.vrp[0, 1] + self.vpn[0, 1] * self.extent[2] * 0.5)],
                        [0, 0, 1, -1 * (self.vrp[0, 2] + self.vpn[0, 2] * self.extent[2] * 0.5)],
                        [0, 0, 0, 1]])

        Rxyz = np.matrix([[self.u[0, 0], self.u[0, 1], self.u[0, 2], 0.0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0.0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

        r1 = np.matrix([[math.cos(a1), 0.0, math.sin(a1), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1 * math.sin(a1), 0.0, math.cos(a1), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        r2 = np.matrix([[1.0, 0.0, 0.0, 0.0],
                        [0.0, math.cos(a2), -1 * math.sin(a2), 0.0],
                        [0.0, math.sin(a2), math.cos(a2), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        t1_inverse = np.matrix([[1.0, 0.0, 0.0, (self.vrp[0, 0] + self.vpn[0, 0] * self.extent[2] * 0.5)],
                                [0.0, 1.0, 0.0, (self.vrp[0, 1] + self.vpn[0, 1] * self.extent[2] * 0.5)],
                                [0.0, 0.0, 1.0, (self.vrp[0, 2] + self.vpn[0, 2] * self.extent[2] * 0.5)],
                                [0.0, 0.0, 0.0, 1.0]])

        tvrc = np.matrix([[self.vrp[0, 0], self.vrp[0, 1], self.vrp[0, 2], 1.0],
                          [self.u[0, 0], self.u[0, 1], self.u[0, 2], 0.0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0.0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0.0]])

        tvrc = (t1_inverse * Rxyz.T * r2 * r1 * Rxyz * t1 * tvrc.T).T

        self.vrp = np.matrix(tvrc.tolist()[0][0:3])
        self.u = np.matrix(tvrc.tolist()[1][0:3])
        self.vup = np.matrix(tvrc.tolist()[2][0:3])
        self.vpn = np.matrix(tvrc.tolist()[3][0:3])

        normalize(self.u)
        normalize(self.vup)
        normalize(self.vpn)


if __name__ == '__main__':
    view = View()
    new_view = view.clone()
    print "U:"

    vector = np.matrix([1.0, 2.0, 3.0])
    print vector
    print "\n\n"
    normalize(vector)
    print vector

    print new_view
    print view.build()
    print view.rotateVRC(2.0, 1.0)
