import pytest
import usqc.util as util
import itk
import numpy as np

def test_bounded_region():
    ref_img = itk.image_from_array(np.zeros((50,50)))
    r = itk.ImageRegion[2]()
    r.SetIndex([-10, 20])
    r.SetUpperIndex([30,  49])
    
    z = util.bounded_region(ref_img, r)
    idx1 = z.GetIndex()
    idx2 = z.GetUpperIndex()
    assert idx1[0] == 0
    assert idx1[1] == 20
    assert idx2[0] == 30
    assert idx2[1] == 49
    
