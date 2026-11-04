import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from dataset import QueryLoader, preprocess_target


class TargetTests(unittest.TestCase):
    def test_prediction_inputs_use_only_joint_query_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'height.npy', np.zeros((256, 256), dtype=np.float32))
            np.save(root / 'beam.npy', np.full((128, 128), -150, dtype=np.float32))
            query = {'height': 'height.npy', 'beam': 'beam.npy',
                     'supervision': {'target': 'nonexistent_target.npy'}}
            condition, beam = QueryLoader(root).inputs([query], 'cpu', arm='A')
            self.assertEqual(tuple(condition.shape), (1, 6, 256, 256))
            self.assertEqual(tuple(beam.shape), (1, 1, 256, 256))
            self.assertEqual(int(condition[0, 0].sum()), 1)
            self.assertEqual(float(condition[0, 0, 127, 127]), 1.)
            self.assertTrue(torch.allclose(condition[:, 2:3], beam))
            self.assertTrue(torch.all(condition[:, 3:] == 0))

    def test_invalid_values_are_not_interpolated_into_training_support(self):
        x=torch.full((1,1,128,128),-120.,dtype=torch.float32)
        x[0,0,20,20]=1000.; x[0,0,40,40]=-300.
        y,valid=preprocess_target(x)
        self.assertEqual(tuple(y.shape),(1,1,256,256))
        self.assertTrue(torch.allclose(y[valid],torch.full_like(y[valid],.6),atol=1e-6))
        self.assertFalse(valid[0,0,40,40]); self.assertFalse(valid[0,0,80,80])
        self.assertTrue(torch.isfinite(y).all())


if __name__ == '__main__': unittest.main()
