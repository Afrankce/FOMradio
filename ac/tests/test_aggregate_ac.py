import unittest
import numpy as np
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import aggregate_test_ac as aggregate
from aggregate_test_ac import scene_means,paired_bootstrap


class AggregateTests(unittest.TestCase):
    def test_scenes_are_equal_weighted_even_with_unequal_records(self):
        rows=[{'scene':'u1','v':0.},{'scene':'u1','v':2.},{'scene':'u2','v':10.}]
        means=scene_means(rows,'v')
        self.assertEqual(means,{'u1':1.,'u2':10.})
        self.assertEqual(float(np.mean(list(means.values()))),5.5)

    def test_bootstrap_is_scene_paired_and_deterministic(self):
        a={42:{'u1':1.,'u2':2.,'u3':3.},43:{'u1':4.,'u2':5.,'u3':6.}}
        c={s:{k:v+2 for k,v in values.items()} for s,values in a.items()}
        r=paired_bootstrap(a,c,repetitions=1000,seed=3)
        self.assertEqual(r['mean_delta_c_minus_a'],2.)
        self.assertEqual(r['ci_95'],[2.,2.])
        self.assertEqual(r,paired_bootstrap(a,c,repetitions=1000,seed=3))

    def test_bootstrap_rejects_misaligned_scene_keys(self):
        with self.assertRaises(ValueError):paired_bootstrap({42:{'u1':1}},{42:{'u2':2}},repetitions=10)

    def test_end_to_end_group_checks_and_macro(self):
        with TemporaryDirectory() as name:
            root=Path(name);old_root,old_groups=aggregate.ROOT,aggregate.GROUPS
            try:
                aggregate.ROOT=root;aggregate.GROUPS=('array',)
                spec={'test_output_root':'results','seed_deviation':'two seeds',
                      'groups_expected_queries':{'array':160}}
                (root/'test_run_spec.json').write_text(json.dumps(spec))
                (root/'manifests').mkdir()
                manifest=[{'query':{'scene':f'u{i}','angle':0,'array':32}} for i in range(160)]
                (root/'manifests'/'array.json').write_text(json.dumps(manifest))
                for arm in ('A','C'):
                    for seed in (42,43):
                        folder=root/'results'/f'{arm}_seed{seed}';folder.mkdir(parents=True)
                        run=root/'runs'/f'{arm}_seed{seed}';run.mkdir(parents=True)
                        (run/'best.pt').write_bytes(f'{arm}{seed}'.encode())
                        digest=aggregate.file_hash(run/'best.pt')
                        ident={'checkpoint_sha256':digest,'files_sha256':{}}
                        (folder/'identity.json').write_text(json.dumps(ident))
                        (folder/'progress.json').write_text(json.dumps({'phase':'complete','completed_predictions':640}))
                        (run/'best.json').write_text(json.dumps({'checkpoint_sha256':digest,'step':5000}))
                        rows=[]
                        for i in range(160):
                            value=float(i)+(2. if arm=='C' else 0.)
                            row={'scene':f'u{i}','angle':0,'array':32,'mae_db':value,
                                 'rmse_db':value,'psnr_db':40.,'ssim':.8,'boundary_mae_db':value,
                                 'outside_range_fraction':0.,'valid_pixels':100,
                                 'valid_ssim_windows':50,'boundary_valid_pixels':20}
                            if arm=='C':row['mu_mae_db']=value+1
                            rows.append(row)
                        for draw in range(4):
                            payload={'arm':arm,'seed':seed,'group':'array','draw':draw,
                                     'identity_sha256':aggregate.stable_hash(ident),'checkpoint_step':5000,'rows':rows}
                            (folder/f'array_draw{draw}.json').write_text(json.dumps(payload))
                with contextlib.redirect_stdout(io.StringIO()):aggregate.main()
                summary=json.loads((root/'results'/'summary.json').read_text())
                paired=summary['groups']['array']['mae_paired_bootstrap']
                self.assertAlmostEqual(paired['mean_delta_c_minus_a'],2.)
                self.assertEqual(paired['ci_95'],[2.,2.])
            finally:
                aggregate.ROOT,aggregate.GROUPS=old_root,old_groups


if __name__=='__main__':unittest.main()
