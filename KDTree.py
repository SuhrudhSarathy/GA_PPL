#!/usr/bin/env python3

from shapely.geometry import Point
from sklearn.neighbors import KDTree
import numpy as np
class Tree():

    '''
        Class for KDTRee
    '''
    def __init__(self, array, leaf_size):
        self.array = array
        self.leaf_size = leaf_size
        self.tree = KDTree(self.array, leaf_size=self.leaf_size)


    def get_nearest_neighbors(self, n, point):
        dist, ind = self.tree.query(point, k = n)
        nn = [self.array[i] for i in ind][0]
        return nn
