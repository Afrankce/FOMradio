import math
import unittest
import numpy as np
from metrics_ac import score_map, boundary_from_height


class RadioMapMetricTests(unittest.TestCase):
    def test_perfect_prediction_and_zero_error(self):
        gt=np.full((32,32),-120.,dtype=np.float32)
        valid=np.ones_like(gt,dtype=bool);boundary=np.zeros_like(valid);boundary[4:8,4:8]=True
        m=score_map(gt,gt,valid,boundary)
        self.assertEqual(m['mae_db'],0.);self.assertEqual(m['rmse_db'],0.)
        self.assertIsNone(m['psnr_db']);self.assertTrue(m['psnr_infinite'])
        self.assertAlmostEqual(m['ssim'],1.,places=10)
        self.assertEqual(m['boundary_mae_db'],0.)

    def test_constant_error_in_db(self):
        gt=np.full((32,32),-120.,dtype=np.float32);pred=gt+10
        valid=np.ones_like(gt,dtype=bool);boundary=np.ones_like(valid)
        m=score_map(pred,gt,valid,boundary)
        self.assertEqual(m['mae_db'],10.);self.assertEqual(m['rmse_db'],10.)
        self.assertAlmostEqual(m['psnr_db'],10*math.log10(300**2/100),places=10)
        self.assertAlmostEqual(m['ssim'],(2*(-120)*(-110)+9)/((-120)**2+(-110)**2+9),places=10)

    def test_invalid_pixels_do_not_enter_metrics_or_ssim_windows(self):
        gt=np.full((32,32),-120.,dtype=np.float32);pred=gt+10
        valid=np.ones_like(gt,dtype=bool);valid[14:18,14:18]=False;gt[~valid]=1000
        m=score_map(pred,gt,valid,valid)
        self.assertEqual(m['mae_db'],10.)
        self.assertAlmostEqual(m['ssim'],(2*(-120)*(-110)+9)/((-120)**2+(-110)**2+9),places=9)
        valid[:]=False;valid[:3,:3]=True
        m=score_map(pred,gt,valid,valid)
        self.assertIsNone(m['ssim']);self.assertEqual(m['valid_ssim_windows'],0)

    def test_boundary_uses_any_building_in_2x2_and_radius3(self):
        height=np.zeros((256,256),dtype=np.float32);height[100,100]=5
        boundary=boundary_from_height(height)
        self.assertEqual(boundary.shape,(128,128))
        self.assertFalse(boundary[50,50]);self.assertTrue(boundary[47,47]);self.assertFalse(boundary[46,46])
        self.assertEqual(int(boundary.sum()),48)

    def test_ssim_matches_skimage_on_fully_valid_windows(self):
        from scipy.ndimage import binary_erosion
        from skimage.metrics import structural_similarity
        rng=np.random.default_rng(7)
        gt=rng.uniform(-250,-30,(40,40));pred=gt+rng.normal(0,5,(40,40))
        valid=np.ones((40,40),bool);valid[12:18,12:18]=False
        m=score_map(pred,gt,valid,valid)
        _,full=structural_similarity(gt,pred,data_range=300.,gaussian_weights=True,
                sigma=1.5,use_sample_covariance=False,win_size=11,full=True)
        support=binary_erosion(valid,structure=np.ones((11,11),bool),border_value=0)
        self.assertAlmostEqual(m['ssim'],float(full[support].mean()),places=12)


if __name__=='__main__':unittest.main()
