import itk
import numpy as np
import random
import typing
import re
import json
import pickle
from pathlib import Path
import os
from glob import glob
import matplotlib.pyplot as plt
import pkgutil
from usqc.util import overlay, get_iterator_setup, add_to_group
from usqc.data import get_files

'''
Label-to-pixel value key for all possible segmentation labels in the CIRS 040GSE phantom.

Note, not all labels will be in a given image, and 3D Slicer reorders the pixel values,
so we need to map each image back to the master SLICER_KEY
"slicer_label_name" : master_label_value

Example values:
"A 1.3" : anechoic 1.3mm target
"AR A1" : the A1-labeled target in the axial resolution cluster
"D1 C -3" : the -3 db contrast target at the first depth
"D1 C H" : the hyperechoic contrast target at the first depth
"D1 E 10" : the 10 kpa elastography target at the first depth
"VH" : a vertical, horizontal, or nearfield wire target
"VH E" : a vertical, horizontal, or nearfield target viewed in the elevational profile
"D1 C H E" : the hyperechoic contrast target at the first depth, view in elevational profile
"background" : no target present

Note: the elevational profile causes circular targets to look like flat ellipses as the
probe has been turned by 45 degrees.
'''
SLICER_KEY = {
    "A 1.3": 1,
    "A 10.0": 6,
    "A 2.0": 2,
    "A 3.0": 3,
    "A 4.5": 4,
    "A 6.7": 5,
    "AR A1": 22,
    "AR A2": 23,
    "AR A3": 24,
    "AR A4": 25,
    "AR A5": 26,
    "AR A6": 27,
    "AR A7": 28,
    "AR B1": 29,
    "AR B2": 30,
    "AR B3": 31,
    "AR B4": 32,
    "AR B5": 33,
    "AR B6": 34,
    "AR C1": 35,
    "AR C2": 36,
    "AR C3": 37,
    "AR C4": 38,
    "AR C5": 39,
    "AR C6": 40,
    "AR D1": 41,
    "AR D2": 42,
    "AR D3": 43,
    "AR D4": 44,
    "AR D5": 45,
    "D1 C -3.0": 9,
    "D1 C -6.0": 8,
    "D1 C -9.0": 7,
    "D1 C 3.0": 10,
    "D1 C 6.0": 12,
    "D1 C -3": 9,
    "D1 C -6": 8,
    "D1 C -9": 7,
    "D1 C 3": 10,
    "D1 C 6": 12,
    "D1 C H": 13,
    "D1 E 10": 46,
    "D1 E 40": 47,
    "D1 E 60": 48,
    "D2 C -3": 16,
    "D2 C -6": 15,
    "D2 C -9": 14,
    "D2 C 3": 17,
    "D2 C 6": 18,
    "D2 C H": 19,
    "D2 E 10": 49,
    "D2 E 40": 50,
    "D2 E 60": 51,
    "VH": 20,
    "VH E": 21,
    "D1 C H E": 52,
    "D1 E 10 E" : 53,
    "background" : 0
}

'''
Pixel value-to-label key for all possible segmentation labels in the CIRS 040GSE phantom.
'''
SLICER_KEY_REV = dict()
for k, v in SLICER_KEY.items():
    SLICER_KEY_REV[v] = k

class CircularTarget():

    @classmethod
    def split_by_value(cls, elements, value=0.5):
        '''
        Returns two lists: elements <= value and elements > value, i.e., the CircularTarget.intensity 

        Used to group elements by inside value.  Useful for deciding factor values in narrowband
        registration by determining which elements are brighter or darker than background.

        Parameters
        ----------
        elements : list of CircularTarget
        value : float

        Returns
        -------
        leq_elements : list of CircularTarget
        gt_elements : list of CircularTarget
        '''

        leq = []
        gt = []
        for x in elements:
            if x.intensity <= value:
                leq.append(x)
            else:
                gt.append(x)
        return leq, gt

    def __init__(self, name, radius, center, intensity, meta=None, actual_radius=None):
        '''
        Parameters
        ----------
        name : str
            Unique name for element, to be used in querying
        radius : float
            size of circle to draw (mm)
        center : ndarray[2]
            physical location [x,y] (mm)
        intensity : float 
            grayscale intensity 0.0-1.0
        meta :dict
            meta data
        actual_radius : float or None
            Specify if the radius to draw is different than the real world radius (e.g. 80 micron wires are too small to register), otherwise, this will equal radius
        '''
        self.name = name
        self.radius = radius
        self.center = center
        self.intensity = intensity
        self.meta = dict() if meta is None else meta
        self.actual_radius = radius if actual_radius is None else actual_radius
        self.spatial_object = itk.EllipseSpatialObject[2].New(RadiusInObjectSpace=1)
        ellipse_transform = itk.AffineTransform[itk.D, 2].New()
        ellipse_transform.Scale([radius, radius])
        ellipse_transform.Translate([center[0], center[1]])
        self.spatial_object.SetObjectToWorldTransform(ellipse_transform)
        self.spatial_object.SetDefaultInsideValue(self.intensity)
        self.spatial_object.Update()

    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in sorted(self.__dict__)))    

    def inside(self, origin, size):
        '''
        Returns true if the entire circular element is within the box specified.

        Parameters
        ----------
        origin : ndarray[2]
            Upper-left corner of box in physical coordinates
        size : ndarray[2]
            Size in physical coordinates of box

        Returns
        -------
        bool
        '''
        # Returns true if the entire circular element is within the box
        # with specified origin and size.

        def point_inside(p, origin, size):
            return p[0] > origin[0] and p[0] < origin[0] + size[0] and \
                   p[1] > origin[1] and p[1] < origin[1] + size[1]

        c = self.center
        r = self.radius
        cardinals = [
            [c[0] - r, c[1]],
            [c[0], c[1] + r],
            [c[0] + r, c[1]],
            [c[0], c[1] - r]
        ]
        for c in cardinals:
            if not point_inside(c, origin, size):
                return False
        return True

class Phantom():
    '''
    Represents the schematic (ideal) form of the CIRS 040GSE phantom. 

    Note, the CIRS 040GSE is a quasi-3D structure.  Most elements are circular wires or tubes
    running through it.  The main exception is the "stepped" anechoic cylinders.  There are five
    steppings, so we have represented this phantom as five 2D "slices" corresponding to each
    stepping of the anechoic elements.

    This object has phantom elements stored as itk.SpatialObjects (via the CircularTarget class)
    and can be used to generate label images.

    Also note, the origin of the phantom has been chosen as the top of the vertical wire group.
    '''
    # OK
    # 7" x 5" x 8" phantom
    # could do full 3D, but for now, just 5 slices (corresponding to the anechoic cylinder steps)
    # circular target = base for intensity scanning
    # elastic target = probably not used for registration but allow for elasticity parameter to be stored
    # axial resolution clusters 1, 2, and 3
    # gray scale clusters 1 and 2 (each with 6 circles)
    # 6 anechoic pairs of 2 (x 5 slices)
    # near field wire points of 5
    # 2 horizontal distance clusters (1 of 6 pts, 1 of 7 pts)
    # vertical distance cluster of 16 pts
    # note, two wires overlap (are the same wire) corresponding to vertical and horizontal
    def __init__(self, make_images=False):
        wire_radius_1_actual = 0.15
        wire_radius_1 = 3 * wire_radius_1_actual
        tmp = np.arange(0, 16)
        self.vertical = [ CircularTarget(f'vertical {y}', wire_radius_1, [0.0, x], 1.0, actual_radius=wire_radius_1_actual) for x, y in zip(10.0 * tmp + 10.0, tmp) ]

        tmp = np.arange(0, 6)
        self.horizontal_4cm = [ CircularTarget(f'horizontal_4cm {y}', wire_radius_1, [x, 40.0], 1.0, actual_radius=wire_radius_1_actual) for x, y in zip(10 * tmp + -30.0, tmp) ]
        tmp = np.arange(0, 7)
        self.horizontal_9cm = [ CircularTarget(f'horizontal_9cm {y}', wire_radius_1, [x, 90.0], 1.0, actual_radius=wire_radius_1_actual) for x, y in zip(20 * tmp + -80.0, tmp) ]

        # confirm the -30.0
        self.nearfield = [ CircularTarget(f'nearfield {x}', wire_radius_1, [-30.0 + 6.0*x, 1.0 + x], 1.0, actual_radius=wire_radius_1_actual) for x in np.arange(0, 5) ]

        wire_radius_2_actual = 0.12
        wire_radius_2 = 3 * wire_radius_2_actual

        a1_1 = np.array([20.0, 30.0])
        a7_1 = a1_1 + np.array([10.75, 0])
        self.resolution_cluster_1 = [
            CircularTarget('ar1 a7', wire_radius_2, a7_1, 1.0, actual_radius=wire_radius_2_actual), # A7
            CircularTarget('ar1 a6', wire_radius_2, a7_1 + [-0.25, 0], 1.0, actual_radius=wire_radius_2_actual), # A6
            CircularTarget('ar1 a5', wire_radius_2, a7_1 + [-0.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A5
            CircularTarget('ar1 a4', wire_radius_2, a7_1 + [-1.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A4
            CircularTarget('ar1 a3', wire_radius_2, a7_1 + [-3.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A3
            CircularTarget('ar1 a2', wire_radius_2, a7_1 + [-6.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A2
            CircularTarget('ar1 a1', wire_radius_2, a7_1 + [-10.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A1
            CircularTarget('ar1 b6', wire_radius_2, a7_1 + [-0.125, -4], 1.0, actual_radius=wire_radius_2_actual), # B6
            CircularTarget('ar1 b5', wire_radius_2, a7_1 + [-0.625, -3], 1.0, actual_radius=wire_radius_2_actual), # B5
            CircularTarget('ar1 b4', wire_radius_2, a7_1 + [-1.625, -2], 1.0, actual_radius=wire_radius_2_actual), # B4
            CircularTarget('ar1 b3', wire_radius_2, a7_1 + [-3.625, -1], 1.0, actual_radius=wire_radius_2_actual), # B3
            CircularTarget('ar1 b2', wire_radius_2, a7_1 + [-6.625, -0.5], 1.0, actual_radius=wire_radius_2_actual), # B2
            CircularTarget('ar1 b1', wire_radius_2, a7_1 + [-10.625, -0.25], 1.0, actual_radius=wire_radius_2_actual) # B1
        ]

        a1_2 = np.array([20.0, 65.0])
        a7_2 = a1_2 + np.array([10.75, 0])
        self.resolution_cluster_2 = [
            CircularTarget('ar2 a7', wire_radius_2, a7_2, 1.0, actual_radius=wire_radius_2_actual), # A7
            CircularTarget('ar2 a6', wire_radius_2, a7_2 + [-0.25, 0], 1.0, actual_radius=wire_radius_2_actual), # A6
            CircularTarget('ar2 a5', wire_radius_2, a7_2 + [-0.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A5
            CircularTarget('ar2 a4', wire_radius_2, a7_2 + [-1.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A4
            CircularTarget('ar2 a3', wire_radius_2, a7_2 + [-3.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A3
            CircularTarget('ar2 a2', wire_radius_2, a7_2 + [-6.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A2
            CircularTarget('ar2 a1', wire_radius_2, a7_2 + [-10.75, 0], 1.0, actual_radius=wire_radius_2_actual), # A1
            CircularTarget('ar2 b6', wire_radius_2, a7_2 + [-0.125, -4], 1.0, actual_radius=wire_radius_2_actual), # B6
            CircularTarget('ar2 b5', wire_radius_2, a7_2 + [-0.625, -3], 1.0, actual_radius=wire_radius_2_actual), # B5
            CircularTarget('ar2 b4', wire_radius_2, a7_2 + [-1.625, -2], 1.0, actual_radius=wire_radius_2_actual), # B4
            CircularTarget('ar2 b3', wire_radius_2, a7_2 + [-3.625, -1], 1.0, actual_radius=wire_radius_2_actual), # B3
            CircularTarget('ar2 b2', wire_radius_2, a7_2 + [-6.625, -0.5], 1.0, actual_radius=wire_radius_2_actual), # B2
            CircularTarget('ar2 b1', wire_radius_2, a7_2 + [-10.625, -0.25], 1.0, actual_radius=wire_radius_2_actual) # B1
        ]

        c1 = np.array([20.0, 105.0])
        c6 = c1 + np.array([15.0, 0])
        self.resolution_cluster_3 = [
            CircularTarget('ar3 c6', wire_radius_2, c6, 1.0, actual_radius=wire_radius_2_actual), # C6
            CircularTarget('ar3 c5', wire_radius_2, c6 + [-1.0, 0], 1.0, actual_radius=wire_radius_2_actual), # C5
            CircularTarget('ar3 c4', wire_radius_2, c6 + [-3.0, 0], 1.0, actual_radius=wire_radius_2_actual), # C4
            CircularTarget('ar3 c3', wire_radius_2, c6 + [-6.0, 0], 1.0, actual_radius=wire_radius_2_actual), # C3
            CircularTarget('ar3 c2', wire_radius_2, c6 + [-10.0, 0], 1.0, actual_radius=wire_radius_2_actual), # C2
            CircularTarget('ar3 c1', wire_radius_2, c6 + [-15.0, 0], 1.0, actual_radius=wire_radius_2_actual), # C1
            CircularTarget('ar3 d5', wire_radius_2, c6 + [-0.75, -5.0], 1.0, actual_radius=wire_radius_2_actual), # D5
            CircularTarget('ar3 d4', wire_radius_2, c6 + [-2.75, -4.0], 1.0, actual_radius=wire_radius_2_actual), # D4
            CircularTarget('ar3 d3', wire_radius_2, c6 + [-5.75, -3.0], 1.0, actual_radius=wire_radius_2_actual), # D3
            CircularTarget('ar3 d2', wire_radius_2, c6 + [-9.75, -2.0], 1.0, actual_radius=wire_radius_2_actual), # D2
            CircularTarget('ar3 d1', wire_radius_2, c6 + [-14.75, -1.0], 1.0, actual_radius=wire_radius_2_actual) # D1
        ]

        elasticity_intensity = 0.4
        self.elasticity_1_5cm = [
            CircularTarget('elasticity_1_5cm 10.0', 3.0, [10.0, 15.0], elasticity_intensity, meta={ 'kPa' : 10.0 }), # in kPa
            CircularTarget('elasticity_1_5cm 40.0', 3.0, [24.0, 15.0], elasticity_intensity, meta={ 'kPa' : 40.0 }),
            CircularTarget('elasticity_1_5cm 60.0', 3.0, [38.0, 15.0], elasticity_intensity, meta={ 'kPa' : 60.0 })
        ]

        self.elasticity_5cm = [
            CircularTarget('elasticity_5cm 10.0', 4.0, [10.0, 50.0], elasticity_intensity, meta={ 'kPa' : 10.0 }),
            CircularTarget('elasticity_5cm 40.0', 4.0, [24.0, 50.0], elasticity_intensity, meta={ 'kPa' : 40.0 }),
            CircularTarget('elasticity_5cm 60.0', 4.0, [38.0, 50.0], elasticity_intensity, meta={ 'kPa' : 60.0 })
        ]

        self.hyperechoic = [
            CircularTarget('contrast_3cm h', 4.0, [-10.0, 30.0], 1.0),
            CircularTarget('contrast_11.5cm h', 5.0, [-10.0, 115.0], 1.0)
        ]

        grayscale_spacing = 12.0
        grayscale_spacing2 = grayscale_spacing * 5.0/4.0
        self.grayscale_3cm = [
            CircularTarget('contrast_3cm 6.0', 4.0, [-10.0 - grayscale_spacing*1, 30.0], 0.8, meta={ 'dB' : 6.0 }),
            CircularTarget('contrast_3cm 3.0', 4.0, [-10.0 - grayscale_spacing*2, 30.0], 0.6, meta={ 'dB' : 3.0 }),
            CircularTarget('contrast_3cm -3.0', 4.0, [-10.0 - grayscale_spacing*3, 30.0], 0.4, meta={ 'dB' : -3.0 }),
            CircularTarget('contrast_3cm -6.0', 4.0, [-10.0 - grayscale_spacing*4, 30.0], 0.2, meta={ 'dB' : -6.0 }),
            CircularTarget('contrast_3cm -9.0', 4.0, [-10.0 - grayscale_spacing*5, 30.0], 0.1, meta={ 'dB' : -9.0 }),
        ]

        self.grayscale_11_5cm = [
            CircularTarget('contrast_11.5cm 6.0', 5.0, [-10.0 - grayscale_spacing2*1, 115.0], 0.8, meta={ 'dB' : 6.0 }),
            CircularTarget('contrast_11.5cm 3.0', 5.0, [-10.0 - grayscale_spacing2*2, 115.0], 0.6, meta={ 'dB' : 3.0 }),
            CircularTarget('contrast_11.5cm -3.0', 5.0, [-10.0 - grayscale_spacing2*3, 115.0], 0.4, meta={ 'dB' : -3.0 }),
            CircularTarget('contrast_11.5cm -6.0', 5.0, [-10.0 - grayscale_spacing2*4, 115.0], 0.2, meta={ 'dB' : -6.0 })
        ]

        # anechoic depths 1.5, 4.5, 7, 10, 13, 16
        # anechoic diameters (@ 1.5/4.5) 1.3, 2.0, 3.0, 4.5, 6.7
        # anechoic diameters (@ 7, 10, 13, 16) 2.0, 3.0, 4.5, 6.7, 10.0
        # -55.0, -38.0
        # 6 slices, 3 slices per attenuation side
        # but middle two are indistinguishable so 5 total
        anechoic_intensity = 0.0
        def make_anechoic(r1, r2, r3, r4):
            return [
                CircularTarget(f'anechoic_1.5cm L {r1*2}', r1, [-55.0, 15.0], anechoic_intensity),
                CircularTarget(f'anechoic_1.5cm R {r2*2}', r2, [-38.0, 15.0], anechoic_intensity),
                CircularTarget(f'anechoic_4.5cm L {r1*2}', r1, [-55.0, 45.0], anechoic_intensity),
                CircularTarget(f'anechoic_4.5cm R {r2*2}', r2, [-38.0, 45.0], anechoic_intensity),
                CircularTarget(f'anechoic_7.0cm L {r3*2}', r3, [-55.0, 70.0], anechoic_intensity),
                CircularTarget(f'anechoic_7.0cm R {r4*2}', r4, [-38.0, 70.0], anechoic_intensity),
                CircularTarget(f'anechoic_10.0cm L {r3*2}', r3, [-55.0, 100.0], anechoic_intensity),
                CircularTarget(f'anechoic_10.0cm R {r4*2}', r4, [-38.0, 100.0], anechoic_intensity),
                CircularTarget(f'anechoic_13.0cm L {r3*2}', r3, [-55.0, 130.0], anechoic_intensity),
                CircularTarget(f'anechoic_13.0cm R {r4*2}', r4, [-38.0, 130.0], anechoic_intensity),
                CircularTarget(f'anechoic_16.0cm L {r3*2}', r3, [-55.0, 160.0], anechoic_intensity),
                CircularTarget(f'anechoic_16.0cm R {r4*2}', r4, [-38.0, 160.0], anechoic_intensity)
            ]
        anechoic_1 = make_anechoic(0.65, 3.35, 1.0, 5.0)
        anechoic_2 = make_anechoic(1.0, 2.25, 1.5, 3.35)
        anechoic_3 = make_anechoic(1.5, 1.5, 2.25, 2.25)
        anechoic_4 = make_anechoic(2.25, 1.0, 3.35, 1.5)
        anechoic_5 = make_anechoic(3.35, 0.65, 5.0, 1.0)
        self.anechoics = [anechoic_1, anechoic_2, anechoic_3, anechoic_4, anechoic_5]

        def make_phantom(anechoic):
            '''
            TODO: make a phantom or slice class that handles referencing the CircularTargets etc better
            ''' 
            ans = itk.GroupSpatialObject[2].New()
            add_to_group(ans, self.vertical)
            add_to_group(ans, self.horizontal_4cm)
            add_to_group(ans, self.horizontal_9cm)
            add_to_group(ans, self.nearfield)

            add_to_group(ans, self.elasticity_1_5cm)
            add_to_group(ans, self.elasticity_5cm)

            add_to_group(ans, self.resolution_cluster_1)
            add_to_group(ans, self.resolution_cluster_2)
            add_to_group(ans, self.resolution_cluster_3)

            add_to_group(ans, self.hyperechoic)
            add_to_group(ans, self.grayscale_3cm)
            add_to_group(ans, self.grayscale_11_5cm)
            add_to_group(ans, anechoic)
            return ans

        self.__phantom_objects = [make_phantom(a) for a in self.anechoics]
        if make_images:
            self.__phantom_images = [self.make_phantom_image(p) for p in self.__phantom_objects]
        else:
            self.__phantom_images = [None for p in self.__phantom_objects]

    def make_phantom_image(self, phantom_object):
        reference_image = itk.Image[itk.F, 2].New()

        ref_origin = np.array([-85.0, 0.0])
        ref_extent = np.array([45.0, 170.0])
        ref_spacing = np.array([0.02, 0.02])
        ref_size = ((ref_extent - ref_origin) / ref_spacing).astype('int')
        reference_image.SetOrigin(ref_origin)
        reference_image.SetSpacing(ref_spacing)
        reference_image.SetRegions(itk.ImageRegion[2](ref_size.tolist()))
        reference_image.Allocate()

        phantom_background = 0.5
        f = itk.SpatialObjectToImageFilter[itk.SpatialObject[2],
                                           itk.Image[itk.F,2]].New(OutsideValue=phantom_background,
                                                                   Input=phantom_object)
        f.SetOrigin(reference_image.GetOrigin())
        f.SetSpacing(reference_image.GetSpacing())
        f.SetSize(reference_image.GetLargestPossibleRegion().GetSize())
        f.SetUseObjectValue(True)
        f.Update()
        return f.GetOutput()

    def get_slice_count(self):
        return len(self.__phantom_objects)

    def get_slice(self, i):
        if i < 0 or i >= self.get_slice_count():
            raise IndexError("Phantom object index out of range")
        return self.__phantom_objects[i]

    def get_image(self, i):
        if i < 0 or i >= self.get_slice_count():
            raise IndexError("Image index out of range")
        if self.__phantom_images[i] is None:
            self.__phantom_images[i] = self.make_phantom_image(self.__phantom_objects[i])
        return self.__phantom_images[i]

    def get_nearfield_target(self, i):
        if i < 0 or i >= len(self.nearfield):
            raise IndexError("Nearfield target query out of range")
        return self.nearfield[i]

    def get_vertical_target(self, i):
        if i < 0 or i >= len(self.vertical):
            raise IndexError("Vertical target query out of range")
        return self.vertical[i]

    def get_horizontal_target_at_4cm(self, i):
        if i < 0 or i >= len(self.horizontal_4cm):
            raise IndexError("Horizontal target query out of range")
        return self.horizontal_4cm[i]

    def get_horizontal_target_at_9cm(self, i):
        if i < 0 or i >= len(self.horizontal_9cm):
            raise IndexError("Horizontal target query out of range")
        return self.horizontal_9cm[i]

    def get_grayscale_target_at_3cm(self, i):
        if i < 0 or i >= len(self.grayscale_3cm):
            raise IndexError("Grayscale target query out of range")
        return self.grayscale_3cm[i]

    def get_grayscale_target_at_11_5cm(self, i):
        if i < 0 or i >= len(self.grayscale_11_5cm):
            raise IndexError("Grayscale target query out of range")
        return self.grayscale_11_5cm[i]

    def get_top_hyperechoic_target(self):
        return self.hyperechoic[0]

    def get_bottom_hyperechoic_target(self):
        return self.hyperechoic[1]

    def get_top_elasticity_target(self, i):
        if i < 0 or i >= len(self.elasticity_1_5cm):
            raise IndexError("Elasticity target query out of range")
        return self.elasticity_1_5cm[i]

    def get_bottom_elasticity_target(self, i):
        if i < 0 or i >= len(self.elasticity_5cm):
            raise IndexError("Elasticity target query out of range")
        return self.elasticity_5cm[i]

    def get_top_resolution_cluster_target(self, i):
        if i < 0 or i >= len(self.resolution_cluster_1):
            raise IndexError("Resolution cluster target query out of range")
        return self.resolution_cluster_1[i]

    def get_middle_resolution_cluster_target(self, i):
        if i < 0 or i >= len(self.resolution_cluster_2):
            raise IndexError("Resolution cluster target query out of range")
        return self.resolution_cluster_2[i]

    def get_bottom_resolution_cluster_target(self, i):
        if i < 0 or i >= len(self.resolution_cluster_3):
            raise IndexError("Resolution cluster target query out of range")
        return self.resolution_cluster_3[i]

    def get_anechoic_target(self, slice_, i):
        if slice_ < 0 or slice_ >= len(self.anechoics):
            raise IndexError("Phantom slice query out of range")
        anechoics = self.anechoics[slice_]
        if i < 0 or i >= len(anechoics):
            raise IndexError("Anechoic target query out of range")
        return anechoics[i]

    def get_element_groups(self, slice_):
        return [
          self.vertical,
          self.horizontal_4cm,
          self.horizontal_9cm,
          self.nearfield,
          self.elasticity_1_5cm,
          self.elasticity_5cm,
          self.resolution_cluster_1,
          self.resolution_cluster_2,
          self.resolution_cluster_3,
          self.hyperechoic,
          self.grayscale_3cm,
          self.grayscale_11_5cm,
          self.anechoics[slice_]
        ]

    def get_elements(self, slice_):
        return [item for group in self.get_element_groups(slice_) for item in group]

    @classmethod
    def as_dict(cls, elements):
        '''
        Convenience method for converting list of elements to dict keyed by name

        Parameters
        ----------
        elements : list of CircularTarget

        Returns
        -------
        dict { str : CircularTarget }
        '''
        return { e.name : e for e in elements }

    def get_keys(self, slice_):
        '''
        Returns list of keys associated with a particular slice

        Parameters
        ----------
        slice_ : int
            Slice of phantom to return element keys of

        Returns
        -------
        list of str
        '''
        return Phantom.as_dict(self.get_elements(slice_)).keys()

    def get_random_element(self, slice_):
        return random.choice(self.get_elements(slice_))

    def get_physical_region(self, slice_, center, radius, multiple=2.0):
        assert multiple > 1.0

        origin = [center[0] - multiple * radius, center[1] - multiple * radius]
        size = [2.0 * multiple * radius, 2.0 * multiple * radius]
        return (origin, size)

    def get_image_region(self, slice_, center, radius, multiple=2.0):
        assert multiple > 1.0
        image = self.get_image(slice_)
        (origin, size) = self.get_physical_region(slice_, center, radius, multiple)
        tl = image.TransformPhysicalPointToIndex(origin)
        br = image.TransformPhysicalPointToIndex([origin[0] + size[0], origin[1] + size[1]])
        size = itk.Index[2]([br[0] - tl[0], br[1] - tl[1]])
        return itk.ImageRegion[2](tl, size)

    def extract_region_image(self, slice_, center, radius, multiple=2.0):
        desired_region = self.get_image_region(slice_, center, radius, multiple)

        ImageType = itk.Image[itk.F, 2]
        extract_filter = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
        extract_filter.SetInput(self.get_image(slice_))
        extract_filter.SetRegionOfInterest(desired_region)
        extract_filter.Update()
        return extract_filter.GetOutput()

    def elements_inside(self, slice_, origin, size):
        '''
        Returns a list of CircularTarget elements completely inside the defined bounding box.

        Parameters
        ----------
        slice_ : int
            Which anechoic slice to return elements from (1 thru 5)
        origin : ndarray[2]
            Upper left corner of bounding box, in physical coordinates
        size : ndarray[2]
            Size of bounding box, in physical coordinates

        Returns
        -------
        list of CircularTarget
        '''
        elements = []

        # Check vertical targets
        for g in self.get_element_groups(slice_):
          for e in g:
            if e.inside(origin, size):
              elements.append(e)
        return elements

    def get_nearest_element(self, slice_, pt, transform=None):
        '''
        Returns the phantom element closest to pt with optional transform.
        
        Parameters
        ----------
        slice_ : int
            Slice of the phantom
        pt : ndarray[2]
            Point in physical space
        transform : None or itk.Transform
            Optional transform to apply to pt before finding the element, e.g.,
            using a registration transform
            
        Returns
        -------
        CircularTarget
        '''
        pt = pt if transform is None else np.array(transform.TransformPoint(pt))
        min_dist = np.Inf
        ans = None
        elems = self.get_elements(slice_)
        for e in elems:
            d = np.linalg.norm(e.center - pt)
            if d < min_dist:
                min_dist = d
                ans = e
        
        return ans
    
    def get_nearest_elements(self, slice_, pts, transform=None):
        '''
        Returns a the nearest phantom element per pt in pts
        
        Parameters
        ----------
        slice_ : int
            Slice of phantom
        pts : list of ndarray[2]
            List of points in physical space
        transform : None or itk.Transform
            Optional transform to apply per pt before finding the element, e.g.,
            using a registration transform
        
        See also
        --------
        get_nearest_element
        
        Returns
        -------
        CircularTarget
        '''
        return [ self.get_nearest_element(slice_, pt, transform) for pt in pts ]
    
    def elements_inside_transform(self, slice_, img, transform):
        '''
        Returns elements that are contained in the img field-of-view after being mapped by transform

        Parameters
        ----------
        slice_ : int
            The slice of phantom to retreive elements from
        img : itk.Image[,2]
            The img defining a bounding box
        transform : itk.Transform
            This transform will map pixel of img into the phantom physical space

        Returns
        -------
        list of CircularTarget
        '''
        r = img.GetLargestPossibleRegion()

        ip1 = img.TransformIndexToPhysicalPoint(r.GetIndex())
        ip2 = img.TransformIndexToPhysicalPoint(r.GetUpperIndex())
        pp1 = transform.TransformPoint(ip1)
        pp2 = transform.TransformPoint(ip2)
        size = np.array(pp2) - np.array(pp1)

        return self.elements_inside(slice_, pp1, size)
              
    def load_feature_map(self, f):
        '''
        Load the image, connected component image, and features corresponding to f.
        
        Will load the preprocessed image and provide a mapping data structure to phantom features.
        
        Parameters
        ----------
        f : str
            File path corresponding to the data point (e.g., to be passed to usqc.data.get_files)
        
        Returns
        -------
        img : itk.Image
            The preprocessed image
        component_img : itk.Image
            The connected component image
        feature_map : dict
            dict with format { connected_component_pixel_value : dict }
        slice_ : int or None
            The corresponding slice in the phantom if it could be guessed, None otherwise
        '''
        
        files = get_files(f)
        
        img = itk.imread(files['preprocessed'])
        with open(files['annotated_distance_map_points'], 'rb') as fp:
            points = pickle.load(fp)
        
        # maps img to phantom, but we don't have an inverse implemented
        trans = itk.transformread(files['registered_transform'])[0]
        
        component_img = itk.imread(files['annotated_component'])
        
        slice_ = self._detect_slice(points, trans)
        s = slice_ if slice_ is not None else 0
        for k, v in points.items():
            center = v['center']
            elem = self.get_nearest_element(s, center, trans)
            v['phantom_label'] = elem.name
            v['phantom_element'] = elem
            
        return img, component_img, points, slice_
        
    def get_feature_by_phantom_label(self, fm, phantom_label):
        '''
        Returns the item in fm with matching phantom_label

        Parameters
        ----------
        fm : dict
            E.g, from phantom.load_feature_map()
        phantom_label : str

        Returns
        -------
        component_label : int
        feature : dict

        Raises
        ------
        KeyError
        '''

        x = [ i for i in fm.items() if i[1]['phantom_label'] == phantom_label ]
        assert len(x) < 2, f'Only one match to "{phantom_label}" expected'
        if len(x) == 0:
            raise KeyError(phantom_label)
        return x[0]
        
    def _detect_slice(self, points, trans):
        '''
        Attempt to detect what slice is represented or return None
        
        This works by checking whether an anechoic element is in points (the only thing differentiating slices)
        and then determining if it is the left or right anechoic element using slice 0 as a reference.  If there
        is an anechoic element, we then determine the slice by its size and whether it was left or right.
        
        Parameters
        ----------
        points : dict
            E.g., from files['annotated_distance_map_points']
        trans : itk.Transform
            E.g., from files['registered_transform']
            
        Returns
        -------
        int or None
        '''
        ans = None
        for v in points.values():
            slicer_label = SLICER_KEY_REV[v['label']]
            if slicer_label.startswith('A '):
                # we ignore the precise size of the anechoic element annotated in order to find the nearest
                # element in slice 0.  _detect_slice_by_match will correct the slice 0 guess with the appropriate
                # slice
                slice_, phantom_label = self._detect_slice_by_match(slicer_label, self.get_nearest_element(0, v['center'], trans))
                if slice_ is not None:
                    ans = slice_
                    break
        return ans
        
    def _detect_slice_by_match(self, slicer_label, guess_elem):
        '''
        Attempts to detect which slice this image is from relying on whether an anechoic feature
        at 1.5cm was manually annotated.
        
        Parameters
        ----------
        slicer_label : str
            The manual annotation label
        guess_elem : CircularTarget
            An element that matches in slice 0.  The algorithm works if a anechoic element in slice
            0 matches, it then corrects the guess.
        
        Returns
        -------
        slice_ : int or None
        phantom_label : str or None
        '''
        
        left_map = {
            'A 1.3' : (0, 'anechoic_1.5cm L 1.3'),
            'A 2.0' : (1, 'anechoic_1.5cm L 2.0'),
            'A 3.0' : (2, 'anechoic_1.5cm L 3.0'),
            'A 4.5' : (3, 'anechoic_1.5cm L 4.5'),
            'A 6.7' : (4, 'anechoic_1.5cm L 6.7')
        }
        
        right_map = {
            'A 1.3' : (4, 'anechoic_1.5cm R 1.3'),
            'A 2.0' : (3, 'anechoic_1.5cm R 2.0'),
            'A 3.0' : (2, 'anechoic_1.5cm R 3.0'),
            'A 4.5' : (1, 'anechoic_1.5cm R 4.5'),
            'A 6.7' : (0, 'anechoic_1.5cm R 6.7')
        }    
        
        if guess_elem.name.startswith('anechoic_1.5cm L'):
            return left_map[slicer_label]
        elif guess_elem.name.startswith('anechoic_1.5cm R'):
            return right_map[slicer_label]
        else:
            return None, None
        
def _read_slicer_segmentation(f):
    '''
    Return a 2D label image from the 3D format Slicer saves as.

    This function correctly remaps the local pixel values of labels in the specified segmentation file
    to the SLICER_KEY values.  Note, the way label meta-data is stored in Slicer is in the header data
    of a .seg.nrrd file.  Slicer stores the label names, and gives labels pixel values in the order they
    are entered in the UI.  So, we need to remap the local pixel values to the standard pixel values
    defined in SLICER_KEY.

    For 2D images, Slicer will add a 3rd dimension (of size 1) to the segmentation file.  This collapses
    that extra dimension.

    For 3D video, e.g., for signal-to-noise or snr measurements, the z-slice used for the manual segmentation
    is arbitrary (and could be in the middle).  Since we clamp the probe while making an snr measurement, we
    will only segment one frame and apply it the entire video.  So, this method will find which z-slice in the
    3D Slicer segmentation file has the labels in it, and will return the corresponding 2D label image.  Note,
    the labels must be in a single slice.  It's easy while segmenting to accidently scroll to a new slice,
    thereby erroneously spreading labels over multiple slices.

    Parameters
    ----------
    f : str
        Path to file (.seg.nrrd).

    Returns
    -------
    itk.Image[,2]
    dict : { str : int }
    '''
    img = itk.imread(f)
    label_map = _slicer_label_meta_to_map(dict(img))
    r = img.GetLargestPossibleRegion()
    idx = r.GetIndex()
    size = r.GetSize()

    if size[2] > 1: # this is an snr video
        # for snr video we only segment a single frame since the probe is physically clamped/immobile
        # slicer default is middle of image, but we will confirm location of segmentation
        maxes = [np.max(img[x,:,:]) for x in range(img.shape[0])]
        npidx = int(np.argwhere(maxes).squeeze()[()])
        idx[2] = idx[2] + npidx
    else: # single slicer 3D image
        idx[2] = 0

    size[2] = 0
    r.SetIndex(idx)
    r.SetSize(size)
    f = itk.ExtractImageFilter[type(img), itk.Image[itk.template(img)[1][0], 2]].New(Input=img, ExtractionRegion=r)
    f.SetDirectionCollapseToSubmatrix()
    f.Update()
    return f.GetOutput(), label_map

def _slicer_label_meta_to_map(meta_dict):
    '''
    Parses names in meta_dict to get {label_name : pixel_value} dict

    Returns
    -------
    dict{str : int}
    '''
    ans = dict()
    regex = r'Segment(?P<seg_id>\d+)_Name'
    for k in meta_dict.keys():
        m = re.match(regex, k)
        if m is not None:
            i = m.group('seg_id')
            ans[meta_dict[f'Segment{i}_Name']] = int(meta_dict[f'Segment{i}_LabelValue'])
    return ans

def _remap_slicer_segmentation(img, img_label_map, master_key=SLICER_KEY):
    '''
    Remaps the pixel values in img to the corresponding values in master_key

    Parameters
    ----------
    img : itk.Image[,2]
    img_label_map : dict {str : int}
    master_key : dict {str : int}

    Returns
    -------
    itk.Image[,2]
    '''
    ans = itk.image_duplicator(img)

    value_map = { 0 : master_key['background'] }
    for key, value in img_label_map.items():
        value_map[value] = master_key[key]

    idx, j0, k0, jn, kn = get_iterator_setup(ans)
    j = j0
    while j < jn:
        k = k0
        while k < kn:
            idx.SetElement(0,j)
            idx.SetElement(1,k)
            ans.SetPixel(idx, value_map[img.GetPixel(idx)])
            k += 1
        j += 1

    return ans

def label_image_to_center_of_mass(component_img, label_img, index_space=False):
    '''
    Computes the center of mass for each label (unique pixel value) in img.

    Assumes something like a connected components filter and small object removal has occurred.

    Parameters
    ----------
    img : itk.Image[,2]
    index_space : bool
        If true, returns the center of mass in index space, physical space otherwise

    Returns
    -------
    dict{int : ndarray[2]}
        Per unique non-zero pixel value in img, value : center of mass
    '''
    idx, j0, k0, jn, kn = get_iterator_setup(component_img)
    tmp = dict()

    # TODO: add label image map

    j = j0
    while j < jn:
        k = k0
        while k < kn:
            idx.SetElement(0, j)
            idx.SetElement(1, k)
            v = component_img.GetPixel(idx)
            if v != 0:
                cur = np.array([j,k])
                d = tmp.get(v)
                if d is None:
                    tmp[v] = { 'center' : cur, 'n' : 1, 'label' : label_img.GetPixel(idx) }
                else:
                    n = d['n']
                    old = d['center']
                    d['center'] = (cur + n*old)/(n+1.0)
                    d['n'] = n + 1.0
            k += 1
        j += 1

    for k, d in tmp.items():
        if not index_space:
            d['center'] = np.array(component_img.TransformContinuousIndexToPhysicalPoint(itk.ContinuousIndex[itk.D,2](d['center'])))

    return tmp

def preprocess_slicer_segmentation(f):
    '''
    Preprocesses 3D Slicer segmentation and saves output.

    Parameters
    ----------
    f : str
        Path to *.seg.nrrd file

    Returns
    -------
    None
    '''
    img1, label_map = _read_slicer_segmentation(f)
    img2 = _remap_slicer_segmentation(img1, label_map)

    fil = itk.ConnectedComponentImageFilter.New(Input=img2)
    fil.Update()
    component_img = fil.GetOutput()
    
    # TODO: update this file format once ITK issue with cast_image not being wrapped for types is
    # resolved.  Want this saved as itk.ULL as itk.UC is probably too small for a connected component
    # image if a segmentation or something similiar goes awry (could quickly make more than 255
    # component clusters)
    component_img = component_img.astype(itk.UC)
    centers_dict = label_image_to_center_of_mass(component_img, img2)
    centers_list = [i['center'] for k, i in centers_dict.items()]

    distmap = points_to_distancemap_filter(centers_list, img2)
    files = get_files(f)
    os.makedirs(Path(files['annotated']).parent, exist_ok=True)
    itk.imwrite(img2, files['annotated'], compression=True)
    itk.imwrite(component_img, files['annotated_component'], compression=True)
    itk.imwrite(distmap, files['annotated_distance_map'], compression=True)
    with open(files['annotated_distance_map_points'], 'wb') as fp:
        pickle.dump(centers_dict, fp)

def points_to_distancemap_filter(pts, label_img):
    '''
    Takes a list of ndarrys or itk.PointSetCollection and returns the distance map image.

    Parameters
    ----------
    pts : list of ndarray[2] or itk.PointSetCollection
    label_img : itk.Image[,2]
        Used as reference image (taking the origin, spacing, size) for the output
    Returns
    -------
    itk.Image[itk.D,2]
    '''

    if type(pts) == list:
        pts2 = itk.PointSet[itk.D,2].New()

        points = pts2.GetPoints()
        for i in range(len(pts)):
            points.InsertElement(i, itk.Point[itk.D,2](pts[i]))
        pts = pts2

    f = itk.PointSetToImageFilter[type(pts), itk.Image[itk.D,2]].New(Input=pts)
    f.SetOrigin(label_img.GetOrigin())
    f.SetSpacing(label_img.GetSpacing())
    f.SetSize(label_img.GetLargestPossibleRegion().GetSize())

    # both Danielssson and PointSetToImageFilter fail to preserves all the specificities of the input image coordinate space
    # so we'll manually correct here
    f2 = itk.DanielssonDistanceMapImageFilter.New(Input=f.GetOutput())
    f2.SetUseImageSpacing(True)
    f2.Update()
    ans = f2.GetOutput()
    ans.SetSpacing(label_img.GetSpacing())
    ans.SetOrigin(label_img.GetOrigin())
    ans.SetDirection(label_img.GetDirection())
    ans.SetRegions(label_img.GetLargestPossibleRegion())

    return ans
    
