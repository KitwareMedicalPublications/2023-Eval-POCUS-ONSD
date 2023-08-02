import itk
import math
import numpy as np
from usqc.data import get_files
import usqc.phantom
import json
from usqc.util import clamped_region
from glob import glob
import usqc.phantom
import usqc.data
import pickle
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pdb


class PSFSample:
    def __init__(self, xs, ys, pts, peaks, peaks_dict, xs_centered):
        '''
        Parameters
        ----------
        xs : list of float
            1D physical coordinates of signal samples (negative values left and positive right of zero-center)
        ys : list of float
            Signal values
        pts : ndarray 2 x N or None
            2D image physical coordinates of samples.  A mean PSFSample doesn't have these and can be None
        peaks : list of int
            Index of peak in signal
        peaks_dict : dict
            I.e. from scipy.signal.find_peaks
        xs_centered : list of float
            1D physical coordinates of signal samples shifted so that zero corresponds to self.peak
        '''
        self._xs = xs
        self._ys = ys
        self._pts = pts
        self._peaks = peaks
        self._peaks_dict = peaks_dict
        self._xs_centered = xs_centered
    
    @property
    def width(self):
        '''
        Peak width in physical units
        '''
        return self._peaks_dict['widths'][0] * (self._xs[1] - self._xs[0])
    
    @property
    def height(self):
        return self._ys[self.peak]
    
    @property
    def peak(self):
        return self._peaks[0]

    @property
    def middle_peak(self):
        return int(np.round((self._peaks_dict['right_ips'][0] + self._peaks_dict['left_ips'][0])/2))
    
    @property
    def xs_centered_middle(self):
        return self._xs - self._xs[self.middle_peak]
    
    @property
    def xs(self):
        return self._xs
    
    @property
    def ys(self):
        return self._ys
    
    @property
    def xs_centered(self):
        return self._xs_centered
    
    @property
    def pts(self):
        return self._pts
    
    @staticmethod
    def _sample_image(center, theta, r, img, num_samples):
        num_samples = num_samples if num_samples is not None else int(np.ceil(2.0 * r / img.GetSpacing()[0]))
        
        v = r * np.array([np.cos(theta), np.sin(theta)])
        pts = np.linspace(center+v, center-v, num=num_samples)
    
        # we want xs to be distance from the center with negative values towards theta and postive values -theta
        xs = np.linalg.norm(center - pts, axis=1)
        xs[0:int(len(xs)/2)] *= -1

        ys = [ img.GetPixel(img.TransformPhysicalPointToIndex(p)) for p in pts ]
        
        return xs, ys, pts
    
    @staticmethod
    def _analyze_peaks(xs, ys, prominence):
        '''
        Finds a singular peak (or raises a ValueError if zero or more than one peak is found)
        
        Parameters
        ----------
        xs : list of float
        ys : list of float
        prominence : float
        
        Returns
        -------
        peaks : list of int
            A 1-value array corresponding to the index of the peak in ys
        peaks_dict : dict
            A meta dict containing additional peak information, see scipy.signal.find_peaks
        xs_centered : list of float
            The x coordinates of the signal recentered so that 0 occurs at the peak index
        '''
        
        # peaks, peaks_dict = find_peaks(ys, width=0, prominence=prominence)
        # if len(peaks) != 1:
        #     raise ValueError(f'Number of peaks {len(peaks)} != 1, adjust prominence')
          
        peaks, peaks_dict = PSFSample._find_single_peak(ys)
        
        xs_centered = xs - xs[peaks[0]]
        return peaks, peaks_dict, xs_centered
    
    @staticmethod
    def _resample(samples):
        '''
        Takes a list of samples and returns them resampled in a common space.  E.g. take
        the interesection of their x coordinates and resample by the min distance between
        x coordinates of the given samples.  If the samples are in the same space, e.g.,
        have identical x coordinates, then this will return a copy.
        
        Parameters
        ----------
        samples : list of tuple
            List of tuples, each tuple is of the form (xs, ys) where xs is and ndarray
            of x coordinates and ys is an ndarray of y values.
            
        Returns
        -------
        list of tuple
        '''
        x1 = np.max([ s[0][0] for s in samples ])
        x2 = np.min([ s[0][-1] for s in samples ])
        diff = np.min([ s[0][1]-s[0][0] for s in samples])
        n = int((x2 - x1) / diff + 1)
        xs = np.linspace(x1, x2, num=n)
        
        return [ (xs, interp1d(s[0], s[1])(xs)) for s in samples ]
    
    @staticmethod
    def calculate(center, theta, r, img, num_samples=None, prominence=0.3):
        '''
        Creates a PSF sample from the given image, center, angle, radius.
        
        Parameters
        ----------
        center : ndarray[2]
            Physical center to sample from (x,y)
        theta : float
            Angle towards the start point (e.g. 0-index of returned values), 0 to 2pi
        r : float
            Physical 1/2 length of sample
        img : itk.Image or None
            Image to sample intensities from
        num_samples : int or None
            If given, determines number of samples returned
        prominence : float
            Prominence threshold used in peak finding, see scipy.signal.find_peaks
        '''
        xs, ys, pts = PSFSample._sample_image(center, theta, r, img, num_samples)
        peaks, peaks_dict, xs_centered = PSFSample._analyze_peaks(xs, ys, prominence=prominence)
        
        return PSFSample(xs, ys, pts, peaks, peaks_dict, xs_centered)
    
    @staticmethod
    def mean(psfs, prominence=0.3, coordinate='xs'):
        '''
        Takes a list of samples, resamples them into a common space, and then returns the mean PSFSample
        
        Parameters
        ----------
        samples : list of tuple
            List of tuples, each tuple is of the form (xs, ys) where xs is and ndarray
            of x coordinates and ys is an ndarray of y values.
        prominence : float
            Prominence threshold used in peak finding, see scipy.signal.find_peaks
        coordinate : { 'xs', 'xs_centered', 'xs_centered_middle' }
            Which source for x coordinates in each signal to be used for alignment.  'xs' is the
            orginal coordinates, 'xs_centered' is the coorinates centered on the peak, and
            'xs_centered_middle' are the coordinate centered on 'middle_peak', e.g. the index
            between the half-height edges of the peak.
            
        Returns
        -------
        PSFSample
        '''
        assert coordinate in { 'xs', 'xs_centered', 'xs_centered_middle' }
        
        if coordinate == 'xs':
            samples = PSFSample._resample([ (p.xs, p.ys) for p in psfs ])
        elif coordinate == 'xs_centered':
            samples = PSFSample._resample([ (p.xs_centered, p.ys) for p in psfs ])
        else:
            samples = PSFSample._resample([ (p.xs_centered_middle, p.ys) for p in psfs ])
        
        yss = np.array([ s[1] for s in samples ])
        ys = np.mean(yss, axis=0)
        xs = samples[0][0] # all resampled samples share the same x
        peaks, peaks_dict, xs_centered = PSFSample._analyze_peaks(xs, ys, prominence)
        
        return PSFSample(xs, ys, None, peaks, peaks_dict, xs_centered)
    
    @staticmethod
    def _find_single_peak(ys):
        '''
        Optimizes scipy.signal.find_peaks over prominence to return a single peak
        
        Parameters
        ----------
        ys : ndarray
            1D array of signal to find the peaks in
        
        Returns
        -------
        peaks : list of int
            Indices of peaks (see scipy.signal.find_peaks)
        peaks_dict : dict
            Meta data for peaks (see scipy.signal.find_peaks)
        '''
        def foo(x, ys):
            peaks, peaks_dict = find_peaks(ys, width=0, prominence=x)
            return (1-len(peaks))**2
        
        res = scipy.optimize.minimize_scalar(foo, bracket=(0.0, 1.0), args=(ys), method='golden')
        if not res.success or res.fun != 0:
            res = scipy.optimize.minimize_scalar(foo, bounds=(0.0, 1.0), args=(ys), method='bounded')

        if not res.success or res.fun != 0:
            peaks, peaks_dict = find_peaks(ys, width=0, prominence=0.01)
            i = np.argmax([ ys[p] for p in peaks ])
            peaks = [ peaks[i] ]
            peaks_dict = { k : [v[i]] for k, v in peaks_dict.items() }
            return peaks, peaks_dict
        else:
            return find_peaks(ys, width=0, prominence=res.x)
        
    def plot(self, ax=None, coordinate='xs', peak='peak'):
        '''
        Parameters
        ----------
        ax : ax or None
            Plot to ax or create a new figure if None
        '''
        assert coordinate in ['xs', 'xs_centered', 'xs_centered_middle']
        assert peak in ['peak', 'middle_peak']
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
        
        if coordinate == 'xs':
            xs = self.xs
        elif coordinate  == 'xs_centered':
            xs = self.xs_centered
        else:
            xs = self.xs_centered_middle
        
        if peak == 'peak':
            p = self.peak
        else:
            p = self.middle_peak
        
        ax.axvline(xs[p], color='g')
        li = int(self._peaks_dict['left_ips'][0])
        ri = int(self._peaks_dict['right_ips'][0])
        lb = int(self._peaks_dict['left_bases'][0])
        rb = int(self._peaks_dict['right_bases'][0])
        wh = self._peaks_dict['width_heights'][0]
        ax.plot(xs[[li, ri]], [wh, wh], color='r')
        ax.plot(xs[[lb, li]], [self._ys[lb], self._ys[lb]], color='orange')
        ax.plot(xs[[ri, rb]], [self._ys[rb], self._ys[rb]], color='orange')
        ax.plot(xs, self._ys)


def extract_region_image(image, center, radius):
    top_left = [center[0] - 2.0 * radius, center[1] - 2.0 * radius]
    bottom_right = [center[0] + 2.0 * radius, center[1] + 2.0 * radius]
    tl = image.TransformPhysicalPointToIndex(top_left)
    br = image.TransformPhysicalPointToIndex(bottom_right)
    size = itk.Index[2]([br[0] - tl[0], br[1] - tl[1]])
    desired_region = itk.ImageRegion[2](tl, size)

    ImageType = itk.Image[itk.F, 2]
    extract_filter = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
    extract_filter.SetInput(image)
    extract_filter.SetRegionOfInterest(desired_region)
    extract_filter.Update()
    return extract_filter.GetOutput()

def calculate_snr_bounding_box(img, pt0, pt1, y_start=0, y_stop=-3, offset=2.0, width=3.0):
    '''
    Calculate a region 2mm left (normal) of the vertical elements, 3mm wide, and spanning the vertical length
    
    Added the pt1 as a potential sanity check in the future (that the probe isn't too far off from vertical
    with the phantom.  Considered calculating a parallelogram region but that seemed like it could add
    confounding issues (such as the shift).
    
    Note this assumes index == origin
    
    Parameters
    ----------
    img : itk.Image
        The preprocessed image (either 2D or 3D)
    pt0 : ndarray[2]
        center of 'vertical 0', physical space
    pt1 : ndarray[2] 
        center of 'vertical 1', physical space
    y_start : float
        Distance from top (in physical space) to crop
    y_stop : float or None
        If None, use the end of the image, if negative, mm from bottom, if positive, absolute y value
    offset : float
        Distance (mm) to the right of the vertical wire group to start the bounding box
    width : float
        Width (mm) of the bounding box
    
    Returns
    -------
    itk.Image
        Note, this image has the bounding box encoded in its GetLargestPossibleRegion()
    '''
    
    idx = img.GetLargestPossibleRegion().GetIndex()
    uidx = img.GetLargestPossibleRegion().GetUpperIndex()
    
    # in 3D
    pt = img.TransformIndexToPhysicalPoint(idx)
    upt = img.TransformIndexToPhysicalPoint(uidx)
    
    if y_stop is not None:
        if y_stop < 0:
            upt[1] = upt[1] + y_stop
        else:
            upt[1] = y_stop
    
    if img.ndim == 3:
        idx1 = img.TransformPhysicalPointToIndex(np.array([pt0[0]+offset, pt[1]+y_start, pt[2]]))
        idx2 = img.TransformPhysicalPointToIndex(np.array([pt0[0]+offset+width, upt[1], upt[2]]))
        r = itk.ImageRegion[3]()
        r.SetIndex(idx1)
        r.SetUpperIndex(idx2)
    else: # ndim == 2
        idx1 = img.TransformPhysicalPointToIndex(np.array([pt0[0]+offset, pt[1]+y_start]))
        idx2 = img.TransformPhysicalPointToIndex(np.array([pt0[0]+offset+width, upt[1]]))
        r = itk.ImageRegion[2]()
        r.SetIndex(idx1)
        r.SetUpperIndex(idx2)
        
    f = itk.ExtractImageFilter[type(img), type(img)].New(Input=img, ExtractionRegion=r)
    f.Update()
    
    return f.GetOutput()

def calculate_snr(snr_img, method='temporal'):
    '''
    Calculate signal-to-noise.
    
    Per x,y, calculate the mean over time and variance over time.  SNR per x,y is mean / std.
    We then collapse the x,y SNR by row by calculating the row median SNR.  Therefore, we return
    a 1D median SNR array over y.  The median used here as the additional benefit of filtering
    out any Inf values due to a single pixel in a row not changing.
    
    Parameters
    ----------
    snr_img : itk.Image[,3]
        Return value from calculate_snr_bounding_box
    method : str
        Method for SNR calculation.  Temporal returns the mean/noise per vertical row over time.  Static only looks at the first frame.
   
    Returns
    -------
    x : ndarray[N]
        Coordinates in physical space (mm), will start at 0
    snr : ndarray[N]
        SNR values
    signal : ndarray[N]
    noise : ndarray[N]
    
    See also
    --------
    calculate_snr_bounding_box
    '''
    assert method in ['temporal', 'static']
    
    r = snr_img.GetLargestPossibleRegion()
    s = r.GetSize()
    idx = r.GetIndex()
    pt = snr_img.TransformIndexToPhysicalPoint(idx)
    x = np.arange(0, s[1]) * snr_img.GetSpacing()[1] + pt[1]
    
    # normalize the array to have values around 0.5
    inorm = 0.5 - np.mean(snr_img)
    snr_arr = itk.array_from_image(snr_img).copy()
    snr_arr += inorm
    
    if method == 'temporal':
        snr = np.median( np.mean(snr_arr, axis=0) / np.std(snr_arr, axis=0), axis=1)
        signal = np.median(np.mean(snr_arr, axis=0), axis=1)
        noise = np.median(np.std(snr_arr, axis=0), axis=1)
    else: # static
        if snr_arr.ndim == 3: # if we want a 'static' measurement on a video, just look at the first frame
            signal = np.mean(snr_arr[0], axis=1)
            noise = np.std(snr_arr[0], axis=1)
            snr = signal / noise
        else:
            signal = np.mean(snr_arr, axis=1)
            noise = np.std(snr_arr, axis=1)
            snr = signal / noise

    return x, snr, signal, noise

def load_and_calculate_snr(f):
    '''
    Given a data point path, load all the files necessary and compute the signal-to-noise.
    
    Intensity values are scaled to a mean of 0.5 before calculating SNR, signal, and noise.
    
    Parameters
    ----------
    f : str
        Path to a file associated with the data point, i.e., to be passed to usqc.data.get_files()
    
    Returns
    -------
    dict of tuple
        If 3D, 'temporal' and 'static' keys, otherwise just 'static'
        Tuple values are
        x : ndarray[N]
            Vertical distance (in physical space, mm) of the snr calculation
        snr : ndarray[N]
            SNR per row at distance x
        snr_img : itk.Image
        signal : ndarray[N]
        noise : ndarray[N]
    
    See also
    --------
    calculate_snr
    '''
    files = get_files(f)
    
    phantom = usqc.phantom.Phantom()
    img, component_img, fm, slice_ = phantom.load_feature_map(f)
    _, v0 = phantom.get_feature_by_phantom_label(fm, 'vertical 0')
    _, v1 = phantom.get_feature_by_phantom_label(fm, 'vertical 1')
    
    with open(files['annotated_snr'], 'rb') as fp:
        y_start = json.load(fp)['y_start']
    
    snr_img = calculate_snr_bounding_box(img, v0['center'], v1['center'], y_start)
    ans = {}
    if snr_img.ndim == 3:
        x, snr, signal, noise = calculate_snr(snr_img)
        ans['temporal'] = (x, snr, snr_img, signal, noise)
        
    x, snr, signal, noise = calculate_snr(snr_img, method='static')
    ans['static'] = (x, snr, snr_img, signal, noise)
        
    return ans

def get_feature_mask(img, component_img, value, pad):
    '''
    Returns a image cropped around feature and padded
    
    Parameters
    ----------
    img : itk.Image[,2]
        Image to that contains signal, e.g., US
    component_img : itk.Image[,2]
        Connected components corresponding to img
    value : int
        Pixel value for the feature component to crop to
    pad : float
        Padding on the top/bottom, left/right to add around the feature (in mm, physical space)
        
    Returns
    -------
    cropped_img : itk.Image[,2]
    cropped_component : itk.Image[,2]
    '''
    pad = np.floor(1.7 / img.GetSpacing()[0]) # assume isospacing, 1.7mm pad in index coordinates

    npimg = itk.array_from_image(component_img)
    idxs = np.argwhere(npimg == value)
    ymin = np.min(idxs[:,0]) - pad
    ymax = np.max(idxs[:,0]) + pad

    xmin = np.min(idxs[:,1]) - pad
    xmax = np.max(idxs[:,1]) + pad

    r = clamped_region(img, xmin, xmax, ymin, ymax)
    fil1 = itk.ExtractImageFilter.New(ExtractionRegion=r, Input=img)
    fil1.Update()
    ans1 = fil1.GetOutput()
    
    fil2 = itk.ExtractImageFilter.New(ExtractionRegion=r, Input=component_img)
    fil2.Update()
    ans2 = fil2.GetOutput()
    return ans1, ans2, r

def calculate_histogram_metrics(cropped_img, cropped_component, bgvalue=0):
    '''
    Calculates Histogram Metrics (Fisher and gCNR)
    
    Assumes that class1 is !bgvalue and class2 is bgvalue in the component image.
    Fisher's Criterion is (m1 - m2)**2/(var1 + var2) where mi and vari are
    the mean and var intensities corresponding to class i.
    
    Parameters
    ----------
    cropped_img : itk.Image[,2]
        Signal/intensity image
    component_img : itk.Image[,2]
        Connected component image defining the two classes
    bgvalue : int
        Value of background class (class 2) in the component image
        
    Returns
    -------
    fisher : float
        The Fisher's Criterion
    gcnr : float
        gCNR
    '''
    npimg1 = itk.array_from_image(cropped_img)
    npimg2 = itk.array_from_image(cropped_component)
    vals1 = npimg1[npimg2 != bgvalue]
    vals2 = npimg1[npimg2 == bgvalue]
    m1 = np.mean(vals1)
    v1 = np.var(vals1)
    m2 = np.mean(vals2)
    v2 = np.var(vals2)
    
    return (m1 - m2)**2 / (v1 + v2), gcnr(vals1, vals2)

def gcnr(vals1, vals2):
    '''
    Calculates the gCNR of two value arrays.
    
    gCNR = 1 - sum_x(min(h(vals1)[x], h(vals2)[x])), i.e. the overlap between histograms corresponding to vals1 and vals2
    
    Parameters
    ----------
    vals1 : ndarray
    vals2 : ndarray
    
    Returns
    -------
    float
    '''
    
    # we know the input images are encoded to (0.0, 1.0) and were likely 255 pixel values
    #pdb.set_trace()
    h1, edges1 = np.histogram(vals1, range=(0,1.0), bins=256, density=True)
    h1 = h1 * np.diff(edges1)
    h2, edges2 = np.histogram(vals2, range=(0,1.0), bins=256, density=True)
    h2 = h2 * np.diff(edges2)
    return 1.0 - np.sum(np.minimum(h1, h2))

def load_and_calculate_histogram_metrics(f, phantom_label, pad):
    '''
    Loads a data point corresponding to f, and calculates the Fisher's Criterion
    
    Parameters
    ----------
    f : str
        File path to a data point, e.g., passed to usqc.data.get_files()
    phantom_label : str
        Phantom label, e.g. 'contrast_3cm -9.0' of the feature to calculate over
    pad : float
        Size of padding around object (in mm, physical space) to calculate the background
        
    Returns
    -------
    fisher : float
        Fisher Criterion (m1 - m2)**2/(var1 + var2)
    gcnr : float
        gCNR
    '''
    files = get_files(f)
    ph = usqc.phantom.Phantom()
    
    img, component_img, fm, slice_ = ph.load_feature_map(f)
    component_label, feature = ph.get_feature_by_phantom_label(fm, phantom_label)
    
    ans1, ans2, r = get_feature_mask(img, component_img, component_label, pad)
    return calculate_histogram_metrics(ans1, ans2)
        
