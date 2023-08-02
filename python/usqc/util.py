import itk
import numpy as np

def bounded_region(ref_img, r):
    '''
    Returns a region that is the intersection between r and ref_img.GetLargestPossibleRegion()

    Parameters
    ----------
    ref_img : itk.Image
    r : itk.ImageRegion

    Returns
    -------
    itk.ImageRegion
    '''
    r2 = ref_img.GetLargestPossibleRegion()
    idx1 = r.GetIndex()
    idx2 = r2.GetIndex()
    for i in range(len(idx1)):
        idx1[i] = max(idx1[i], idx2[i])

    idx3 = r.GetUpperIndex()
    idx4 = r2.GetUpperIndex()
    for i in range(len(idx3)):
        idx3[i] = min(idx3[i], idx4[i])

    ans = itk.ImageRegion[r.GetImageDimension()]()
    ans.SetIndex(idx1)
    ans.SetUpperIndex(idx3)
    return ans

def bounded_extract_image(img, r):
    '''
    Calls itk.extract_image_filter but ensures that if r is outside of img that it is bounded.

    Parameters
    ----------
    img : itk.Image
        Image to extract from
    r : itk.ImageRegion
        Region to extract which will be clamped to img.GetLargestPossibleRegion()

    Returns
    -------
    itk.Image
    '''
    return itk.extract_image_filter(img, extraction_region=bounded_region(img, r))

def extract_slice(img, s=0, dim=2):
    '''
    Convenience method for removing a slice from a 3D image.

    Parameters
    ----------
    img : itk.Image[,3]
    s : int
        Relative slice (relative to img index) to take
    dim : int
        Dimension to collapse, i.e., retrieve a slice from

    Examples
    --------

    '''
    r = img.GetLargestPossibleRegion()
    idx = r.GetIndex()
    size = r.GetSize()

    size[dim] = 0
    idx[dim] += s
    r.SetIndex(idx)
    r.SetSize(size)
    f = itk.ExtractImageFilter[type(img), itk.Image[itk.template(img)[1][0], 2]].New(Input=img, ExtractionRegion=r)
    f.SetDirectionCollapseToSubmatrix()
    f.Update()
    return f.GetOutput()

def box_to_region(upperleft, size, ref_img):
    '''
    Converts a box defined by upperleft and size in physical coordinate to an itk.ImageRegion

    Parameters
    ----------
    upperleft : ndarray[2]
    size : ndarray[2]
    ref_img : itk.Image[,2]
        Reference image to define physical to index transform (TODO: index is pixel coord of origin in ITK?)

    Returns
    -------
    itk.ImageRegion[2]
    '''
    ans = itk.ImageRegion[2]()
    idx1 = ref_img.TransformPhysicalPointToIndex(upperleft)
    idx2 = ref_img.TransformPhysicalPointToIndex(upperleft + size)
    ans.SetIndex(idx1)
    ans.SetSize(idx2-idx1)
    return ans

def overlay(img, mask, bg_value=None):
    '''
    Creates an RGB overlay image (as an ndarray).

    Parameters
    ----------
    img : itk.Image[,2]
        Image to be overlayed
    mask : itk.Image[,2]
        Label image to overlay
    bg_value : float or None
        Specifies the pixel value to use as background, or uses the median pixel in mask if None

    Returns
    -------
    ndarray
    '''

    bg_value = np.median(mask) if bg_value is None else bg_value
    tmp = np.array(img) # makes a copy
    idx = mask != bg_value
    tmp[idx] = (tmp[idx] + mask[idx])/2.0

    return np.dstack((img, tmp, img))

def get_iterator_setup(img):
    '''
    Returns indices and limits needed to iterate through an image.

    Parameters
    ----------
    itk.Image[,2]

    Returns
    -------
    idx : itk.Index
    j0 : int
    k0 : int
    jn : int
    kn : int

    Examples
    --------
    idx, j0, k0, jn, kn = get_iterator_setup(img)
    j = j0
    while j < jn:
        k = k0
        while k < kn:
            idx.SetElement(0,j)
            idx.SetElement(1,k)
            ans.SetPixel(idx, 255)
            k += 1
        j += 1
    '''
    idx = img.GetLargestPossibleRegion().GetIndex()
    size = img.GetLargestPossibleRegion().GetSize()
    j0 = idx.GetElement(0)
    k0 = idx.GetElement(1)
    jn = j0 + size[0]
    kn = k0 + size[1]
    return idx, j0, k0, jn, kn 

def add_to_group(group, x, copy=True):
    '''
    Add x (single object or list) to group (itk.GroupSpatialObject)

    Convience method that by default copies the objects before adding to the group.

    Parameters
    ----------
    group : itk.GroupSpatialObject
    x : itk.SpatialObject or list of itk.SpatialObject
    copy : bool
        Whether to make a copy of x when adding to group (default is True and is the safe way to use this)
    '''

    if type(x) == list:
        for y in x:
            z = y.spatial_object.Clone() if copy else y.spatial_object
            group.AddChild(z)
    else:
        z = x.spatial_object.Clone() if copy else x.spatial_object
        group.AddChild(x.spatial_object.Copy())

def clamped_region(img, xmin, xmax, ymin, ymax):
    '''
    Returns a region clamped to img.GetLargestPossibleRegion()

    Returns an itk.ImageRegion that is the intersection between img and the provided values

    Parameters
    ----------
    img : itk.Image[,2]
    xmin : int
    Left x index
    xmax : int
    Right x index
    ymin : int
    Top y index
    ymax : int
    Bottom y index

    Returns
    -------
    itk.ImageRegion[2]
    '''
    ans = itk.ImageRegion[2]()
    r = img.GetLargestPossibleRegion()

    x1 = max(xmin, r.GetIndex()[0])
    x2 = min(xmax, r.GetUpperIndex()[0])
    y1 = max(ymin, r.GetIndex()[1])
    y2 = min(ymax, r.GetUpperIndex()[1])

    ans.SetIndex([int(x1), int(y1)])
    ans.SetUpperIndex([int(x2), int(y2)])
    return ans
