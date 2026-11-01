import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import plot_test_examples_ac as plotting


class PlotTest(unittest.TestCase):
    def test_plot_fixed_scenes_and_paired_maps(self):
        with TemporaryDirectory() as name:
            root=Path(name);old=plotting.ROOT
            try:
                plotting.ROOT=root;(root/'test_run_spec.json').write_text(json.dumps({'test_output_root':'results'}))
                selection={'scene_ids':['u1','u80','u160'],'array':32,'angle':0,
                           'selection':'numeric scene order 1,80,160'}
                for arm in ('A','C'):
                    folder=root/'results'/f'{arm}_seed42';(folder/'selected_maps').mkdir(parents=True)
                    (folder/'selection.json').write_text(json.dumps(selection))
                    for scene in selection['scene_ids']:
                        gt=np.full((128,128),-120,dtype=np.float32);valid=np.ones((128,128),bool)
                        np.savez_compressed(folder/'selected_maps'/f'{scene}_draw0.npz',
                                            ground_truth_db=gt,prediction_db=gt+5,valid=valid)
                with contextlib.redirect_stdout(io.StringIO()):plotting.main()
                for ext in ('pdf','png'):
                    self.assertGreater((root/'results'/f'array_examples_seed42_draw0.{ext}').stat().st_size,1000)
            finally:plotting.ROOT=old


if __name__=='__main__':unittest.main()
