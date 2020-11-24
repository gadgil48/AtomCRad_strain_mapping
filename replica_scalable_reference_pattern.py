# Replica of ScalableReferencePattern found in:
# https://github.com/pyxem/pyxem/blob/master/pyxem/components/scalable_reference_pattern.py
# Author: Sanket Gadgil, Date: 24/11/2020

import numpy as np
from hyperspy.component import Component
from replica_tensor_field import DisplacementGradientMap
from skimage import transform as tf


class ScalableReferencePattern(Component):
    '''
    Replica class that defines a version of ScalableReferencePattern to which
    additional fitting parameters can be added.
    '''
    def __init__(
        self, signal2D, d11=1.0, d12=0.0, d21=0.0, d22=1.0, t1=0.0, t2=0.0,
        theta=0.2*np.pi,  # rotational angle
        intensity=1.0,  # intensity multiplier for the diffraction pattern
        order=3
    ):

        Component.__init__(self, [
            "d11", "d12", "d21", "d22", "t1", "t2",
            "theta",
            "intensity",
        ])

        self._whitelist["signal2D"] = ("init,sig", signal2D)
        self.signal = signal2D
        self.order = order
        self.d11.value = d11
        self.d12.value = d12
        self.d21.value = d21
        self.d22.value = d22
        self.t1.value = t1
        self.t2.value = t2

        self.theta.value = theta
        self.intensity.value = intensity

    def function(self, x, y):
        '''
        Function used to define how the reference signal will be deformed.
        This will be used by the fitting algorithm to calculate
        a goodness of fit.

        Returns
        ---------
        transformed : np.ndarray()
            Transformed image that will be used by the fitting algorithm.
        '''
        signal2D = self.signal.data
        order = self.order
        d11 = self.d11.value
        d12 = self.d12.value
        d21 = self.d21.value
        d22 = self.d22.value
        t1 = self.t1.value
        t2 = self.t2.value

        theta = self.theta.value
        intensity = self.intensity.value

        D = np.array([[d11, d12, t1], [d21, d22, t2], [0.0, 0.0, 1.0]])
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]
        ])

        shifty, shiftx = np.array(signal2D.shape[:2])/2

        shift = tf.SimilarityTransform(translation=[-shiftx, -shifty])
        dform = tf.AffineTransform(matrix=D)
        rform = tf.AffineTransform(matrix=R)

        shift_inv = tf.SimilarityTransform(translation=[shiftx, shifty])

        rotate_transform = tf.warp(
            signal2D, rform.inverse, order=order
        )

        transformed = tf.warp(
            rotate_transform, (shift + (dform + shift_inv)).inverse,
            order=order
        )

        transformed *= intensity

        return transformed

    def construct_displacement_gradient(self):
        '''
        Constructs the displacements grids needed to generate strain maps.

        Returns
        -------

        D : TensorField
            Displacement gradient map based on stretch/scale/shear transforms
            of the reference image for each pixel in the signal to be fitted.
        Rot : TensorField
            Displacement gradient map based on rotational transforms of the
            reference image for each pixel in the signal to be fitted.
        '''

        D = DisplacementGradientMap(np.ones(np.append(self.d11.map.shape, (3, 3))))
        Rot = DisplacementGradientMap(np.ones(np.append(self.d11.map.shape, (3, 3))))

        D.data[:, :, 0, 0] = self.d11.map["values"]
        D.data[:, :, 1, 0] = self.d12.map["values"]
        D.data[:, :, 2, 0] = 0.0
        D.data[:, :, 0, 1] = self.d21.map["values"]
        D.data[:, :, 1, 1] = self.d22.map["values"]
        D.data[:, :, 2, 1] = 0.0
        D.data[:, :, 0, 2] = 0.0
        D.data[:, :, 1, 2] = 0.0
        D.data[:, :, 2, 2] = 1.0

        Rot.data[:, :, 0, 0] = np.cos(self.theta.map["values"])
        Rot.data[:, :, 1, 0] = -np.sin(self.theta.map["values"])
        Rot.data[:, :, 2, 0] = 0.0
        Rot.data[:, :, 0, 1] = np.sin(self.theta.map["values"])
        Rot.data[:, :, 1, 1] = np.cos(self.theta.map["values"])
        Rot.data[:, :, 2, 1] = 0.0
        Rot.data[:, :, 0, 2] = 0.0
        Rot.data[:, :, 1, 2] = 0.0
        Rot.data[:, :, 2, 2] = 0.0

        return D, Rot
