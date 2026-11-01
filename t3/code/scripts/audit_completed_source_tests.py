"""Audit existing full tests against final checkpoints and saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.matern_source import digest_state
from training.same_frequency_trainer import _atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();out=root/'completed_test_4groups';torch.set_num_threads(2)
    manifest=[json.loads(x) for x in (root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines()]
    scenes={r['scene_id'] for r in manifest if r['split']=='test'}
    assert len(scenes)==160
    receipt=json.loads((root/'source_snapshot.json').read_text())
    for name,h in receipt.items():assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==h,name
    expected={(s,n,k) for s in scenes for n in (2,4,8,12) for k in range(4)}
    reference={};checks={}
    for name in ('W','S2','S8','MS'):
        run=root/'8x8'/name;data=json.loads((run/'test_diagnostics.json').read_text());identity=data['identity']
        config=json.loads((run/'config.json').read_text())
        payload=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert payload['trainer_state']['completed_epochs']==identity['epoch']==1000
        assert payload['run_identity']['config_sha256']==config['config_sha256']
        assert digest_state(payload['ema'])==identity['ema_sha256']
        del payload
        assert identity['source']==config['source_specification']
        assert not config['internal_noise'] and not config['feedback']
        keys=[(r['scene_id'],r['steps'],r['realization']) for r in data['records']]
        assert len(keys)==len(set(keys))==2560 and set(keys)==expected
        assert data['n_scenes']==160 and identity['realizations']==4
        index={k:r for k,r in zip(keys,data['records'])};max_difference=0.;count=0
        for n in (2,8):
            for k in range(4):
                files=list((run/f'test_diagnostics_predictions/N{n}/seed_{k}').glob('*.npz'))
                assert {f.stem for f in files}==scenes
                for f in files:
                    with np.load(f,allow_pickle=False) as z:
                        pred,target,mask=z['prediction'],z['target'],z['valid_mask']
                        assert pred.shape==target.shape==mask.shape==(1,256,256)
                        assert mask.dtype==bool and mask.any() and np.isfinite(pred).all()
                        error=(np.clip(pred,0,1).astype('float64')-target.astype('float64'))[mask]*300
                        rmse=float(np.sqrt(np.mean(error**2)))
                        difference=abs(rmse-index[f.stem,n,k]['db_rmse']);assert difference<1e-5,(name,f,difference)
                        max_difference=max(max_difference,difference);count+=1
                        h=hashlib.sha256(target.tobytes()+mask.tobytes()).hexdigest()
                        if f.stem in reference:assert reference[f.stem]==h
                        else:reference[f.stem]=h
        checks[name]=dict(epoch=1000,ema_sha256=identity['ema_sha256'],config_sha256=config['config_sha256'],
            test_json_sha256=hashlib.sha256((run/'test_diagnostics.json').read_bytes()).hexdigest(),
            unique_metric_records=len(keys),saved_predictions_checked=count,max_recomputed_rmse_difference=max_difference,
            source_matches_training=True,checkpoint_matches=True)
        print(name,checks[name],flush=True)
    _atomic_json(out/'test_audit.json',dict(status='passed',scope='Existing full test reused; no new model inference',
        sources=checks,scenes=160,training_seeds=[42],steps=[2,4,8,12],realizations=4,
        frozen_files_verified=len(receipt),target_mask_equal_across_groups=True,excluded='S32 training incomplete'))


if __name__=='__main__':main()
