#!/usr/env/python
# Copyright 2019 The Centre for Translational Data Science (CTDS) 
# at the University of Sydney. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""

Created by Created by Hadi Afshar. 
2019-10-22
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
import jax

from sklearn.datasets import make_spd_matrix
from tqdm import tqdm


class Model:
    def u(self, q):
        raise Exception('not implemented')

    def partial(self, q):
        return jax.jacfwd(self.u)(q)

    def first_discontinuity_x_tx_dimx(self, q, p):
        """
        :param q: start location
        :param p: velocity
        :return: (first_x, min_time_x, dim_x) where:
                first_x     is the nearest point where line q(t) = q + p.t reaches the first discontinuity
                min_time_x  is the time to reach the first discontinuity
                dim_x       assuming boundaries are in form q_i=b, the 'i' associated with the collision boundary.

                Note: if by t_max, a discontinuity is not reached, None, np.inf, None is returned.
        """
        raise Exception('not implemented')

    def get_dim(self):
        """
        :return: The dimensionality of the model
        """
        raise Exception('not implemented')

    def model_name(self):
        raise Exception('not implemented')


class CubeLayerModel(Model):
    def __init__(self, dim, bounds):
        """
        :param dim: dimension of the space
        :param bounds: a list of floats. E.g [b_1, ..., b_k] representing k cubes, centered at (0,...,0) and sides b_i*2
            we assume all bounds b_i are positive
        """
        self.dim = dim
        self.pos_sorted_bounds = np.sort(bounds)
        assert self.pos_sorted_bounds[0] > 0  # we assume all bounds (thus the smallest) are positive

    def get_dim(self):
        return self.dim

    def model_name(self):
        return 'cube'

    def u(self, q):
        # assert q.shape == (self.dim,)
        norm_inf = np.max(np.abs(q))
        for i in range(len(self.pos_sorted_bounds)):
            if norm_inf <= self.pos_sorted_bounds[i]:
                return self.value_in_layer(layer=i, q=q)
        return self.value_in_layer(layer=len(self.pos_sorted_bounds), q=q)

    def value_in_layer(self, layer, q):
        assert layer in range(len(self.pos_sorted_bounds) + 1)
        raise Exception('not implemented')

    def bounds_of_layer_surrounding(self, q):
        result = []

        abs_max_q_i = np.max(np.abs(q))

        pos_upper_bound = np.inf
        pos_lower_bound = 0

        for b_index, b in enumerate(self.pos_sorted_bounds):
            if abs_max_q_i < b:
                pos_upper_bound = b
                if b_index > 0:
                    pos_lower_bound = self.pos_sorted_bounds[b_index - 1]
                break
        if pos_upper_bound == np.inf:
            pos_lower_bound = self.pos_sorted_bounds[-1]

        if pos_lower_bound > 0:
            result.extend([-pos_lower_bound, pos_lower_bound])

        if pos_upper_bound < np.inf:
            result.extend([-pos_upper_bound, pos_upper_bound])

        return np.array(result).sort()

    def first_discontinuity_x_tx_dimx(self, q, p):
        # 1. find the lower and upper bounds of the layer we are in:
        layer_bounds = self.bounds_of_layer_surrounding(q)

        # 2. for each dimension, in what time, t, do we meet the lower and upper bounds?
        #       . Is t>0?
        #       . In that collision point is that co-ordination the maximum q_i?
        #    if both conditions hold then we have find a collision point.
        min_time_x = np.inf
        first_x = None
        dim_x = None
        for d in range(self.dim):
            for b in layer_bounds:
                collision_time = (b - q[d]) / p[d]
                if collision_time > 0:
                    collision_point = q + p * collision_time
                    if np.max(np.abs(collision_point)) == np.abs(collision_point[d]):
                        if collision_time < min_time_x:
                            min_time_x = collision_time
                            first_x = collision_point
                            dim_x = d

        return first_x, min_time_x, dim_x


class Model0(CubeLayerModel):
    def __init__(self, dim, bounds=np.array([2.5, 3.2])):
        super().__init__(dim=dim, bounds=bounds)

    def value_in_layer(self, layer, q):
        if layer == 0:
            return 1.0
        if layer == 1:
            return 2.0
        if layer == 2:
            return 3.0
        raise Exception('unexpected region {i}'.format(i=layer))


class Model1(CubeLayerModel):
    def __init__(self, dim, bounds=np.array([2.5, 3.2])):
        super().__init__(dim=dim, bounds=bounds)
        self.a = np.array(make_spd_matrix(n_dim=dim, random_state=1359))  # a random symmetric positive-definite matrix

    def value_in_layer(self, layer, q):
        q_col = np.reshape(q, (1, -1))
        o = q_col.dot(self.a).dot(q_col.T)
        assert o.shape == (1, 1)  # it's a matrix but should be a number
        o = o[0][0] ** 0.5

        if layer == 0:
            return o
        if layer == 1:
            return 1 + o
        if layer == 2:
            return 3000 + o  # np.inf
        raise Exception('unexpected region {i}'.format(i=layer))



class SphereLayerModel(Model):
    def __init__(self, dim, bounds):
        """
        :param dim: dimension of the space
        :param bounds: a list of floats. E.g [b_1, ..., b_k] representing k cubes, centered at (0,...,0) and sides b_i*2
            we assume all bounds b_i are positive
        """
        self.dim = dim
        self.pos_sorted_bounds = np.sort(bounds)
        assert self.pos_sorted_bounds[0] > 0  # we assume all bounds (thus the smallest) are positive

    def get_dim(self):
        return self.dim

    def model_name(self):
        return 'sphere'

    def u(self, q):
        assert q.shape == (self.dim,)

        norm2 = (q.dot(q)) ** 0.5  # sum(abs(x)**ord)**(1./ord) but that is not jax object?!

        for i in range(len(self.pos_sorted_bounds)):
            if norm2 <= self.pos_sorted_bounds[i]:
                return self.value_in_layer(layer=i, q=q)
        return self.value_in_layer(layer=len(self.pos_sorted_bounds), q=q)

    def value_in_layer(self, layer, q):
        assert layer in range(len(self.pos_sorted_bounds) + 1)
        raise Exception('not implemented')

    def first_discontinuity_x_tx_dimx(self, q, p):
        min_time_x = np.inf
        first_x = None

        qp = q.dot(p)
        pp = p.dot(p)
        qq = q.dot(q)
        for radius in self.pos_sorted_bounds:
            '''
                collision times are solutions to:
                p.dot(p) * (t^2) 
                + 2 * (q.dot(p)) * t
                + q.dot(q) - R^2 = 0 
            '''
            delta = qp * qp - pp * (qq - radius ** 2.0)
            if delta >= 0.0:
                # collision(s) exists:
                t1 = (-qp + delta ** 0.5) / pp
                t2 = (-qp - delta ** 0.5) / pp

                # only positive times matter:
                if t1 <= 0:
                    t1 = np.inf
                if t2 <= 0:
                    t2 = np.inf
                collision_time = np.min([t1, t2])

                # for numerical error prevention
                if 0.00001 < collision_time < min_time_x:
                    min_time_x = collision_time
                    first_x = q + p * collision_time

        return first_x, min_time_x, None  # dim_x is not returned



def test_cube1():
    np.set_printoptions(precision=3)
    old_model = Model0(dim=2, bounds=np.array([3.0, 6.0]))  # this works
    model = FastRefCubeModel(dim=2, A_type='bi_value_A')
    import numpy

    q = np.array([10.0, 0.0])
    p = np.array([-1.0, -0.0])
    first_x, min_time_x, dim_x = model.first_discontinuity_x_tx_dimx(q=q, p=p)
    old_first_x, old_min_time_x, old_dim_x = old_model.first_discontinuity_x_tx_dimx(q=q, p=p)
    print('first_x: {},\t\t min_time_x: {},\t\t dim_x: {} '.format(first_x, min_time_x, dim_x))
    print('old_first_x: {},\t old_min_time_x: {},\t old_dim_x: {} '.format(old_first_x, old_min_time_x, old_dim_x))

    # exit()

    collision_q0 = []
    collision_q1 = []
    for i in tqdm(range(2000)):
        q = np.array([numpy.random.uniform(-30.30, 30.30), numpy.random.uniform(-30.30, 30.30)])
        p = np.array([numpy.random.uniform(-10, 10), numpy.random.uniform(-10, 10)])

        first_x, _, _ = model.first_discontinuity_x_tx_dimx(q=q, p=p)

        if first_x is not None:
            collision_q0.append(first_x[0])
            collision_q1.append(first_x[1])

    import matplotlib.pyplot as plt
    plt.scatter(numpy.array(collision_q0),
                numpy.array(collision_q1), alpha=0.5, s=1)
    plt.show()


def main():
    test_cube1()


class FastRefCubeModel(Model):
    def __init__(self, dim, bounds=(3., 6.)):
        assert len(bounds) == 2
        b1 = bounds[0]
        b2 = bounds[1]
        assert 0 < b1 < b2
        self.b1 = b1
        self.b2 = b2
        self.layer_borders = {0: [-b1, b1], 1: [-b2, -b1, b1, b2], 2: [-b2, b2]}
        self.VERY_SMALL_VAL = 0.0001

        self.a = np.array(
                make_spd_matrix(n_dim=dim  # , random_state=2020
                                ))  # a random symmetric positive-definite matrix

        self.dim = dim

    def get_dim(self):
        return self.dim

    def value_in_layer(self, layer, q):
        o = q[None, :].dot(self.a).dot(q[:, None])
        o = o[0][0] ** 0.5

        if layer == 0:
            return o
        if layer == 1:
            return 1 + o
        if layer == 2:
            return np.inf
        raise Exception('unexpected region {i}'.format(i=layer))

    def u(self, q):
        return self.value_in_layer(self.layer(q), q)

    def first_discontinuity_x_tx_dimx(self, q, p):
        l1 = self.layer(q)
        tx_min = np.inf
        dimx = None
        for b in self.layer_borders[l1]:
            # t, q, p are vectors but b is scalar
            t = (b - q) / p  # t are times to hit b in each dimension
            sorted_dims = np.argsort(t)  # dimension/index of collision times sorted
            for i in sorted_dims:  # collision dims
                ti = t[i]
                if self.VERY_SMALL_VAL < ti < tx_min:
                    l2 = self.layer(q + p * (ti + self.VERY_SMALL_VAL))  # region just behind the border
                    if l2 != l1:
                        tx_min = ti
                        dimx = i
        x = q + p * tx_min
        return x, tx_min, dimx

    def layer(self, q):
        v = np.max(np.abs(q))
        if v < self.b1:
            return 0
        elif v < self.b2:
            return 1
        else:
            return 2


if __name__ == "__main__":
    main()
