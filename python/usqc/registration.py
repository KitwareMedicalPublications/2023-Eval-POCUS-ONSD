import re
import numpy as np
import itk
from datetime import datetime
from usqc.phantom import Phantom
import usqc.phantom as ph
from usqc.util import extract_slice, bounded_extract_image, box_to_region, overlay
from pathlib import Path
from usqc.data import get_files
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os

def get_centering_x_translation(fixed, moving):
    '''
    Return the translation shifting moving (x-wise) over to center on fixed (in physical space)

    Parameters
    ----------
    fixed : itk.Image[,2]
    moving : itk.Image[,2]

    Returns
    -------
    ndarray[2]
    '''
    fi1 = np.array(fixed.GetLargestPossibleRegion().GetIndex())
    fi2 = np.array(fi1) + np.array(fixed.GetLargestPossibleRegion().GetSize())
    fi3 = np.array([(fi1[0] + fi2[0])/2.0, fi1[1]])
    idx1 = itk.ContinuousIndex[itk.D,2]()
    idx1[0] = fi3[0]
    idx1[1] = fi3[1]
    fp = fixed.TransformContinuousIndexToPhysicalPoint(idx1)

    mi1 = np.array(moving.GetLargestPossibleRegion().GetIndex())
    mi2 = np.array(mi1) + np.array(moving.GetLargestPossibleRegion().GetSize())
    mi3 = np.array([(mi1[0] + mi2[0])/2.0, mi1[1]])
    
    idx2 = itk.ContinuousIndex[itk.D,2]()
    idx2[0] = mi3[0]
    idx2[1] = mi3[1]
    mp = moving.TransformContinuousIndexToPhysicalPoint(idx2)

    return fp - mp

def get_requested_box(input_img, x_position, x_buffer=5, y_buffer=5):
    '''
    Given an input_img and a guess at the x position of the middle of the image, return a region of the phantom to be used as an atlas.

    Parameters
    ----------
    input_img : itk.Image[,2]
        input_img origin assumed to be (0,0)
    x_position : float
    x_buffer : float
        Additional space on either side of the target region to allow for errors in x_translation (in mm)
    y_buffer : float
        Additional space at the bottom of the target region to allow for errors (in mm)

    Returns
    -------
    upperleft : ndarray[2]
    size : ndarray[2]
    '''
    phys_size = np.array(input_img.GetLargestPossibleRegion().GetSize()) * np.array(input_img.GetSpacing())
    depth = phys_size[1] + y_buffer
    min_x = x_position - (phys_size[0] / 2.0 + x_buffer)

    return np.array([min_x, 0.0]), np.array([phys_size[0] + 2.0*x_buffer, depth])

class PhantomRegistration:
    '''
    Registers an ultrasound image or video of the CIRS 040GSE to the phantom schematic, 
    per usqc.Phantom.

    Currently works in two steps, a translation-based registration, and then a non-rigid
    B-spline registration.  The initial guess to the translation is currently based off
    of the filename with the syntax [image_label]-[replicate].  See init_x_translations
    for image_label values.

    '''
    init_x_translations = {
        'anechoic_1' : (55+38)/2, # centered between anechoic targets, 1.3 left, 6.7 mm right
        'anechoic_2' : (55+38)/2, # 2.0 left, 4.5 mm right
        'anechoic_3' : (55+38)/2, # 3.0 left, 3.0 mm right
        'anechoic_4' : (55+38)/2, # 4.5 left, 2.0 mm right
        'anechoic_5' : (55+38)/2, # 6.7 left, 1.3 mm right
        'contrast_6' : 10, # centered on hyperechoic contrast target (H)
        'contrast_5' : 22, # centered on 6 db contrast target
        'contrast_4' : 34, # centered on 3 db contrast target
        'contrast_3' : 46, # centered on -3 db contrast target
        'gain_1' : 46, # centered on -3 db contrast target
        'gain_2' : 46,
        'gain_3' : 46,
        'gain_4' : 46,
        'gain_5' : 46,
        'gain_snr_1' : 0, # centered on vertical wire targets
        'gain_snr_2' : 0, # centered on vertical wire targets
        'gain_snr_3' : 0, # centered on vertical wire targets
        'gain_snr_4' : 0, # centered on vertical wire targets
        'gain_snr_5' : 0, # centered on vertical wire targets
        'contrast_1_2' : (58+70)/2, # centered between -9 and -6 db contrast targets
        'contrast_2' : 58,
        'contrast_1' : 70,
        'vertical' : 0, # centered on vertical wire targets
        'elevational' : 0, # centered on vertical wire targets, but probe rotated by 45 degrees
        'near' : 15, # centered in middle of nearfield target group
        'axial_lateral' : (-30.75-10.625)/2, # centered on middle of axial resolution target group
        'snr' : 0 # centered on vertical wire targets (as there is an empty space near them)
    }

    def __init__(self, data_dir='../data/CIRS 040GSE'):
        self._data_dir = data_dir
        self.phantom = Phantom()

        self.phantom_images = [itk.imread(str(Path(x))) for x in glob(self._data_dir + '/phantom_image_*.mha')]
        self.phantom_mask = itk.imread(str(Path(self._data_dir + '/phantom_mask.mha')))
        self.phantom_distance = itk.imread(str(Path(self._data_dir + '/phantom_distancemap.mha')))
  
    def _register_translation(self, fixed_dist, moving_dist, moving_mask):
        '''
        Registers moving_dist to fixed_dist using translation.

        Parameters
        ----------
        fixed_dist : itk.Image[,2]
            Distance map derived from a landmark-based pointset, e.g., the ultrasound image
        moving_dist : itk.Image[,2]
            Distance map derived from landmark-based pointset, e.g., the phantom image
        moving_mask : itk.Image[,2]
            Binary mask around landmark points of moving_dist.  If the mask is too large, the
            registration will be too sensitive to distances between points.
        Returns
        -------
        itk.TranslationTransform
            Maps moving_dist to fixed_dist
        float
            Final metric value of registration (lower is better)
        '''
        fixed_image = fixed_dist
        moving_image = moving_dist

        init_transform = itk.TranslationTransform[itk.D, 2].New()
        init_params = init_transform.GetParameters()
        init_params[0] = -get_centering_x_translation(fixed_image, moving_image)[0]
        init_transform.SetParameters(init_params)
        transform = init_transform.Clone()

        # see the ITK Software Guide regarding hand-tuning and choice of 
        # optimization parameters
        # any changes to these registration parameters will affect performance
        # note, there is little justification of these parameters outside of
        # performance
        optimizer = itk.RegularStepGradientDescentOptimizerv4.New()
        optimizer.SetLearningRate(1);
        optimizer.SetMinimumStepLength(0.0001);
        optimizer.SetRelaxationFactor(0.5);

        metric = itk.MeanSquaresImageToImageMetricv4[type(fixed_image), type(moving_image)].New()

        mask_object = itk.ImageMaskSpatialObject[2].New(
            Image=moving_mask)
        mask_object.Update()
        metric.SetMovingImageMask(mask_object)


        registration = itk.ImageRegistrationMethodv4[type(fixed_image), type(moving_image)].New(
            FixedImage=fixed_image,
            MovingImage=moving_image,
            Metric=metric,
            Optimizer=optimizer,
            InitialTransform=transform,
        )

        registration.SetNumberOfLevels(2)
        registration.SetSmoothingSigmasPerLevel([2, 0])
        registration.SetShrinkFactorsPerLevel([1, 1])
        registration.Update()

        trans_transform = registration.GetModifiableTransform()
        return trans_transform, optimizer.GetValue()

    def _register_bspline(self, fixed_dist, moving_dist, moving_mask, trans_transform):
        fixed_image = fixed_dist
        moving_image = moving_dist

        generator = itk.NormalVariateGenerator.New()
        generator.Initialize(12345) # explicit seed to generator

        spline_order = 3
        BTransformType = itk.BSplineTransform[itk.D, 2, spline_order]
        transform = BTransformType.New()

        BInitializerType = itk.BSplineTransformInitializer[BTransformType, type(fixed_image)]
        grid_nodes_count = 12
        mesh_size = itk.Size[2]()
        mesh_size[0] = grid_nodes_count - spline_order
        mesh_size[1] = mesh_size[0]
        initializer = BInitializerType.New()
        initializer.SetTransform(transform)
        initializer.SetImage(fixed_image)
        initializer.SetTransformDomainMeshSize(mesh_size)
        initializer.InitializeTransform()
        transform.SetIdentity()

        metric = itk.MeanSquaresImageToImageMetricv4[type(fixed_image), type(moving_image)].New()

        mask_object = itk.ImageMaskSpatialObject[2].New(
            Image=moving_mask)
        mask_object.Update()
        metric.SetMovingImageMask(mask_object)

        # see the ITK Software Guide regarding hand-tuning and choice of 
        # optimization parameters
        # any changes to these registration parameters will affect performance
        # note, there is little justification of these parameters outside of
        # performance
        optimizer = itk.LBFGSOptimizerv4.New()
        optimizer.SetGradientConvergenceTolerance(0.00005);
        optimizer.SetLineSearchAccuracy(1.2);
        optimizer.SetDefaultStepLength(.05);
        optimizer.TraceOn();
        optimizer.SetMaximumNumberOfFunctionEvaluations(1000);

        r2 = itk.ImageRegistrationMethodv4[type(fixed_image), type(moving_image)].New(
            FixedImage=fixed_image,
            MovingImage=moving_image,
            Metric=metric,
            Optimizer=optimizer,
            InitialTransform=transform,
            MovingInitialTransform=trans_transform,
        )
        r2.SetNumberOfLevels(1)
        r2.SetSmoothingSigmasPerLevel([2])
        r2.SetShrinkFactorsPerLevel([1])
        r2.Update()
        bspline_transform = r2.GetModifiableTransform()
        return bspline_transform, optimizer.GetValue()


    def get_requested_image_by_input_type(self, input_img, phantom_img, phantom_mask, input_type):
        upperleft, size = get_requested_box(input_img, -PhantomRegistration.init_x_translations[input_type])
        r = box_to_region(upperleft, size, phantom_img)
        return bounded_extract_image(phantom_img, r), bounded_extract_image(phantom_mask, r)

    def get_input_type(self, f):
        '''
        Parses out the type of the image from the filename.

        Returns
        -------
        str
            One of ['anechoic', 'contrast_6', 'contrast_5', ..., 'contrast_1_2', 'vertical', 'nearfield', 'axial_lateral', 'snr']
        '''
        return Path(f).stem.split('-')[0]

    # TODO, refactor this to take the segmentation file in as a parameter...
    def register(self, f, trans_transform=None):
        '''
        Registered the phantom to the data corresponding to f.

        This uses the  annotated_distance_map image corresponding to f as the fixed image and the corresponding cropped
        distance map image of the phantom as the moving image.  The final_transform returned is a composite of 
        a translation and a bspline transformation.

        Parameters
        ----------
        f : str
            Path to preprocessed image file to register.

        trans_transform : itk.TranslationTransform[2], optional
            if None, use registration to find the translation transform.  This is used to "fix" registrations where
            the default registration doesn't work.  If specified, trans_metric is returned as 0

        Returns
        -------
        img : itk.Image[,2]
            Ultrasound image
        ph_img : itk.Image[,2]
            Cropped phantom label image corresponding to moving image region
        ph_dist : itk.Image[,2]
            Cropped phantom distance image used as moving image
        ph_mask : itk.Image[,2]
            Cropped phantom mask image used as moving image
        final_transform : itk.CompositeTransform
            0 transform is translation, 1 transform is bspline
        trans_metric : float
            Final value of the translation registration
        bspline_metric : float
            Final value of the bspline nonrigid registration


        Notes
        -----
        f must be properly named (e.g. one of 'anechoic-1')...

        '''

        files = get_files(f)
        img = itk.imread(files['preprocessed'])

        if img.ndim == 3:
            img = extract_slice(img)

        img_label = itk.imread(files['annotated'])
        img_dist = itk.imread(files['annotated_distance_map'])

        # TODO: set the slice number according to whether and which anechoic structure is in the image
        ph_img, _ = self.get_requested_image_by_input_type(img, self.phantom_images[0], self.phantom_mask, self.get_input_type(f))
        ph_dist, ph_mask = self.get_requested_image_by_input_type(img, self.phantom_distance, self.phantom_mask, self.get_input_type(f))

        if trans_transform is None:
            trans_transform, trans_metric = self._register_translation(img_dist, ph_dist, ph_mask)
        else:
            trans_metric = 0   

        bspline_transform, bspline_metric = self._register_bspline(img_dist, ph_dist, ph_mask, trans_transform)

        final_transform = itk.CompositeTransform.New()
        final_transform.AddTransform(trans_transform)
        final_transform.AddTransform(bspline_transform)

        return img, ph_img, ph_dist, ph_mask, final_transform, trans_metric, bspline_metric

    def write(self, f, img, ph_img, ph_dist, ph_mask, final_transform, trans_metric, bspline_metric):
        '''
        Writes the output from register() to disk.

        Parameters
        ----------
        f : str
            The input path used in register().  This is used to compute the output file paths.

        See register() for the explanation of the other parameters.
        '''
        files = get_files(f)
        r_img = resample_image(img, ph_img, final_transform)
        r_dist = resample_image(img, ph_dist, final_transform)
        r_mask = resample_image(img, ph_mask, final_transform)
        overlay_img = itk.image_from_array(overlay(img, r_img))

        mydir = str(Path(files['registered']).parent)
        os.makedirs(mydir, exist_ok=True)

        itk.imwrite(r_img, files['registered'], compression=True)
        itk.imwrite(r_dist, files['registered_distancemap'], compression=True)
        itk.imwrite(r_mask, files['registered_mask'], compression=True)
        itk.imwrite(overlay_img, files['registered_overlay'], compression=True)
        itk.transformwrite([final_transform], files['registered_transform'])
        with open(files['registered_metrics'], 'wb') as fp:
            pickle.dump({'trans_metric' : trans_metric, 'bspline_metric' : bspline_metric}, fp)

def resample_image(fixed_image, moving_image, trans):
    '''
    Convenience method for transforming moving_image onto fixed_image

    Parameters
    ----------
    fixed_image : itk.Image
    moving_image : itk.Image
    trans : itk.Transform

    Returns
    -------
    itk.Image
    '''
    resampler = itk.ResampleImageFilter.New(
        Input=moving_image, 
        Transform=trans, 
        UseReferenceImage=True, 
        ReferenceImage=fixed_image,
        DefaultPixelValue=0)
    resampler.Update()
    return resampler.GetOutput()

