from math import sqrt, pi, cos, sin
from random import random

import numpy as np
from matplotlib import pyplot as plt, animation


def length(vec):
    return sqrt(sum(vec ** 2))


def random_walk(a, r):
    o = np.array([0, 0, 0])  # origin
    rr = length(a)  # radius of sphere
    h = r * sqrt(rr ** 2 - (r / 2) ** 2) / rr  # radius of circle
    doc = sqrt(rr ** 2 - h ** 2)  # distance from o to the circle plane
    for i in range(30):
        oa = a - o  # oa vector
        oc = oa * doc  # oc vector
        alpha = random() * 2 * pi
        obb = np.array([cos(alpha) * r, sin(alpha) * r, h])  # random vector on xoy
        x = oc[0]
        y = oc[1]
        z = oc[2]
        w = sqrt(y ** 2 + z ** 2)
        u = sqrt(x ** 2 + y ** 2 + z ** 2)
        matrix = np.array([[w / u, -x * y / w / u, -x * z / w / u],
                           [0, z / w, -y / w],
                           [x / u, y / u, z / u]])
        inv_mat = np.linalg.inv(matrix)
        cb = inv_mat.dot(obb)
        b = oc + cb
        print(b, length(b), length(cb), matrix)
    return b


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


if __name__ == '__main__':
    random_walk(np.array([0, 0, 1]), 0.2)
