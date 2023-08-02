import math
import pytest
from usqc.phantom import Phantom
import usqc.qc as qc
import usqc.data
from pathlib import Path
import itk
@pytest.fixture(scope="session", autouse=True)
def phantom():
    return Phantom()

@pytest.fixture(scope="session", autouse=False) # it takes forever to make the phantom image
def image(phantom):
    return phantom.get_image(0)

@pytest.fixture()
def data_root():
    return Path(__file__).parent / Path('data')

def percent_diff(expected, actual):
    return 100.0 * abs(actual - expected) / expected

class TestPhantom:
    def test_phantom_construction(self, phantom):
        assert phantom is not None

    def test_nearfield_targets(self, phantom):
        for i in range(5):
            target = phantom.get_nearfield_target(i)
            assert target.radius == pytest.approx(0.45)
            assert target.center == pytest.approx([-30.0 + 6.0 * i, 1.0 +i])
            assert target.intensity == pytest.approx(1.0)
            assert target.actual_radius == pytest.approx(0.15)

        with pytest.raises(IndexError):
            target = phantom.get_nearfield_target(5)

    def test_vertical_targets(self, phantom):
        for i in range(16):
            target = phantom.get_vertical_target(i)
            assert target.radius == pytest.approx(0.45)
            assert target.center == pytest.approx([0.0, 10.0 * i + 10.0])
            assert target.intensity == pytest.approx(1.0)
            assert target.actual_radius == pytest.approx(0.15)

        with pytest.raises(IndexError):
            target = phantom.get_vertical_target(16)

    def test_horizontal_targets_at_4cm(self, phantom):
        for i in range(6):
            target = phantom.get_horizontal_target_at_4cm(i)
            assert target.radius == pytest.approx(0.45)
            assert target.center == pytest.approx([10.0 * i - 30.0, 40.0])
            assert target.intensity == pytest.approx(1.0)
            assert target.actual_radius == pytest.approx(0.15)

        with pytest.raises(IndexError):
            target = phantom.get_horizontal_target_at_4cm(6)


    def test_horizontal_targets_at_9cm(self, phantom):
        for i in range(7):
            target = phantom.get_horizontal_target_at_9cm(i)
            assert target.radius == pytest.approx(0.45)
            assert target.center == pytest.approx([20.0 * i - 80.0, 90.0])
            assert target.intensity == pytest.approx(1.0)
            assert target.actual_radius == pytest.approx(0.15)

        with pytest.raises(IndexError):
            target = phantom.get_horizontal_target_at_9cm(7)

    def test_grayscale_targets_at_3cm(self, phantom):
        intensities = [0.8, 0.6, 0.4, 0.2, 0.1]
        dBs = [6.0, 3.0, -3.0, -6.0, -9.0]
        for i in range(5):
            target = phantom.get_grayscale_target_at_3cm(i)
            assert target.radius == pytest.approx(4.0)
            assert target.center == pytest.approx([-10.0 - 12.0 * (i + 1), 30.0])
            assert target.intensity == pytest.approx(intensities[i])
            assert target.actual_radius == pytest.approx(4.0)
            assert target.meta
            assert target.meta['dB'] == pytest.approx(dBs[i])

        with pytest.raises(IndexError):
            target = phantom.get_grayscale_target_at_3cm(5)

    def test_grayscale_targets_at_11_5cm(self, phantom):
        intensities = [0.8, 0.6, 0.4, 0.2]
        dBs = [6.0, 3.0, -3.0, -6.0]
        for i in range(4):
            target = phantom.get_grayscale_target_at_11_5cm(i)
            assert target.radius == pytest.approx(5.0)
            assert target.center == pytest.approx([-10.0 - 15.0 * (i + 1), 115.0])
            assert target.intensity == pytest.approx(intensities[i])
            assert target.meta
            assert target.meta['dB'] == pytest.approx(dBs[i])

        with pytest.raises(IndexError):
            target = phantom.get_grayscale_target_at_11_5cm(4)

    def test_elements_inside_transform(self, phantom, data_root):
        f = str(data_root / 'test_experiment' / 'preprocessed' / 'butterfly-iq' / 'contrast_6-1.mha')
        files = usqc.data.get_files(f)
        img = itk.imread(files['preprocessed'])
        transform = itk.transformread(files['registered_transform'])[0] # is a one-element list
        elems = phantom.elements_inside_transform(0, img, transform)
        names = [ e.name for e in elems ]
        assert set(names) == set(['vertical 0', 'vertical 1', 'vertical 2', 'nearfield 1', 'nearfield 2', 'nearfield 3', 'nearfield 4', 'contrast_3cm h', 'contrast_3cm 6.0'])
        
            
    def test_elasticity_targets(self, phantom):
        # Top set
        xs = [10.0, 24.0, 38.0]
        kPas = [10.0, 40.0, 60.0]
        for i in range(3):
            target = phantom.get_top_elasticity_target(i)
            assert target.radius == pytest.approx(3.0)
            assert target.center == pytest.approx([xs[i], 15.0])
            assert target.intensity == pytest.approx(0.4)
            assert target.meta
            assert target.meta['kPa'] == pytest.approx(kPas[i])

        with pytest.raises(IndexError):
            target = phantom.get_top_elasticity_target(3)

        # Bottom set
        for i in range(3):
            target = phantom.get_bottom_elasticity_target(i)
            assert target.radius == pytest.approx(4.0)
            assert target.center == pytest.approx([xs[i], 50.0])
            assert target.intensity == pytest.approx(0.4)
            assert target.meta
            assert target.meta['kPa'] == pytest.approx(kPas[i])

        with pytest.raises(IndexError):
            target = phantom.get_bottom_elasticity_target(3)

    def test_as_dict(self, phantom):
        d = phantom.as_dict(phantom.get_elements(0))
        assert len(d) ==  100
        
        fil1 = [k for k in d.keys() if k.startswith('contrast_3cm')]
        assert len(fil1) == 6
        assert 'contrast_3cm h' in fil1
        assert 'contrast_3cm 3.0' in fil1
    
    def test_get_keys(self, phantom):
        keys = phantom.get_keys(0)
        assert len(keys) == 100
        assert 'contrast_3cm h' in keys
        
    def test_resolution_clusters(self, phantom):
        def test_cluster(origin, centers, depth):
            for i in range(len(centers)):
                center = [
                    origin[0] + centers[i][0],
                    origin[1] + centers[i][1]
                ]
                if depth == 0:
                    target = phantom.get_top_resolution_cluster_target(i)
                elif depth == 1:
                    target = phantom.get_middle_resolution_cluster_target(i)
                else:
                    target = phantom.get_bottom_resolution_cluster_target(i)
                assert target.radius == pytest.approx(0.36)
                assert target.center == pytest.approx(center)
                assert target.intensity == pytest.approx(1.0)
                assert not target.meta

            with pytest.raises(IndexError):
                i = len(centers)
                if depth == 0:
                    target = phantom.get_top_resolution_cluster_target(i)
                elif depth == 1:
                    target = phantom.get_middle_resolution_cluster_target(i)
                else:
                    target = phantom.get_bottom_resolution_cluster_target(i)

        # Top
        origin = [30.75, 30.0]
        centers = [
            [0.0, 0.0],
            [-0.25, 0.0],
            [-0.75, 0.0],
            [-1.75, 0.0],
            [-3.75, 0.0],
            [-6.75, 0.0],
            [-10.75, 0.0],
            [-0.125, -4.0],
            [-0.625, -3.0],
            [-1.625, -2.0],
            [-3.625, -1.0],
            [-6.625, -0.5],
            [-10.625, -0.25]
        ]
        test_cluster(origin, centers, depth=0)

        # Middle
        origin = [30.75, 65.0]
        test_cluster(origin, centers, depth=1)

        # Bottom
        origin = [35.0, 105.0]
        centers = [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-3.0, 0.0],
            [-6.0, 0.0],
            [-10.0, 0.0],
            [-15.0, 0.0],
            [-0.75, -5.0],
            [-2.75, -4.0],
            [-5.75, -3.0],
            [-9.75, -2.0],
            [-14.75, -1.0]
        ]
        test_cluster(origin, centers, depth=2)

    def test_hyperechoic(self, phantom):
        # Top
        target = phantom.get_top_hyperechoic_target()
        assert target.radius == pytest.approx(4.0)
        assert target.center == pytest.approx([-10.0, 30.0])
        assert target.intensity == pytest.approx(1.0)
        assert not target.meta

        # Bottom
        target = phantom.get_bottom_hyperechoic_target()
        assert target.radius == pytest.approx(5.0)
        assert target.center == pytest.approx([-10.0, 115.0])
        assert target.intensity == pytest.approx(1.0)
        assert not target.meta

    def test_anechoic_stepped_cylinders(self, phantom):
        radii = [
            [0.65, 3.35, 0.65, 3.35, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0],
            [1.0, 2.25, 1.0, 2.25, 1.5, 3.35, 1.5, 3.35, 1.5, 3.35, 1.5, 3.35],
            [1.5, 1.5, 1.5, 1.5, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25],
            [2.25, 1.0, 2.25, 1.0, 3.35, 1.5, 3.35, 1.5, 3.35, 1.5, 3.35, 1.5],
            [3.35, 0.65, 3.35, 0.65, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0]
        ]
        centers = [
            [-55.0, 15.0],
            [-38.0, 15.0],
            [-55.0, 45.0],
            [-38.0, 45.0],
            [-55.0, 70.0],
            [-38.0, 70.0],
            [-55.0, 100.0],
            [-38.0, 100.0],
            [-55.0, 130.0],
            [-38.0, 130.0],
            [-55.0, 160.0],
            [-38.0, 160.0]
        ]
        for s in range(phantom.get_slice_count()):
            for i in range(12):
                target = phantom.get_anechoic_target(s, i)
                assert target.radius == pytest.approx(radii[s][i])
                assert target.center == pytest.approx(centers[i])
                assert target.intensity == pytest.approx(0.0)
                assert not target.meta

            with pytest.raises(IndexError):
                target = phantom.get_anechoic_target(s, 12)

        with pytest.raises(IndexError):
            target = phantom.get_anechoic_target(phantom.get_slice_count(), 0)

    @pytest.mark.randomize(ncalls=100)
    def test_cylinder_contrast(self, phantom, image):
        # Naive test
        # Choose a random element in the first phantom image
        # Assert that its value (1.0) at cylinder center
        # Is greater than the value two radii away in each cardinal direction
        element = phantom.get_random_element(0)
        contrast = qc.cylinder_contrast(image=image,
                                        center=element.center,
                                        radius=element.radius)
        west = contrast['W']
        assert west[0] > west[-1]
        north = contrast['N']
        assert north[0] > north[-1]
        east = contrast['E']
        assert east[0] > east[-1]
        south = contrast['S']
        assert south[0] > south[-1]

    @pytest.mark.randomize(theta=float, min_num=0.0, max_num=2.0 * math.pi, ncalls=100)
    def test_boundary_estimation_1(self, image, theta):
        expected = 0.65
        actual = qc.estimate_boundary(image=image,
                                      center=[-55.0, 15.0],
                                      radius=expected,
                                      theta=theta)
        assert percent_diff(actual, expected) <= 5.0 # allow 5% error

    @pytest.mark.randomize(slice=int, min_num=0, max_num=4, ncalls=100)
    def test_elements_inside(self, phantom, slice):
        # 1. Pull a random element
        # 2. Draw a slightly larger region around it (1.25x radius)
        # 3. Query phantom for which elements are in the region
        # 4. Expect to get the original element back
        element = phantom.get_random_element(slice)
        (origin, size) = phantom.get_physical_region(slice,
                                                     element.center,
                                                     element.radius,
                                                     multiple=1.25)
        elements_inside = phantom.elements_inside(slice, origin, size)

        # Two of the elements are represented in both the vertical set
        # and one horizontal set
        assert len(elements_inside) == 1 or len(elements_inside) == 2
        for e in elements_inside:
            assert e.center == pytest.approx(element.center)
            assert e.radius == pytest.approx(element.radius)

