import unittest
import torch
from dataset import preprocess_target


class TargetTests(unittest.TestCase):
    def test_invalid_values_are_not_interpolated_into_training_support(self):
        x=torch.full((1,1,128,128),-120.,dtype=torch.float32)
        x[0,0,20,20]=1000.; x[0,0,40,40]=-300.
        y,valid=preprocess_target(x)
        self.assertEqual(tuple(y.shape),(1,1,256,256))
        self.assertTrue(torch.allclose(y[valid],torch.full_like(y[valid],.6),atol=1e-6))
        self.assertFalse(valid[0,0,40,40]); self.assertFalse(valid[0,0,80,80])
        self.assertTrue(torch.isfinite(y).all())


if __name__ == '__main__': unittest.main()
