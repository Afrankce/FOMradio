"""Verify full final tests, prediction arrays, no-EMA identity and replay invariance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.matern_source import digest_state
from scripts.monitor_matern_suite import atomic

VARIANTS=('CASA','ADD','FILM','NSOP')


def audit(root):
    torch.set_num_threads(2)
    out=root/'final_evaluation';out.mkdir(exist_ok=True)
    ref=Path(json.loads((root/'execution_scope.json').read_text())['reference'])
    manifest=[json.loads(s) for s in (root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl').read_text().splitlines()]
    scenes={r['scene_id'] for r in manifest if r['split']=='test'}
    assert len(scenes)==160
    receipt=json.loads((root/'source_snapshot.json').read_text())
    for name,h in receipt.items():assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==h,name
    expected={(s,n,k) for s in scenes for n in (2,4,8,12) for k in range(4)}
    target_hashes={};checks={}
    for v in VARIANTS:
        directory=root/'8x8'/v
        cfg=json.loads((directory/'config.json').read_text())
        data=json.loads((directory/'test_diagnostics.json').read_text())
        identity=data['identity']
        payload=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        old=torch.load(ref/'8x8'/v/'last.pt',map_location='cpu',weights_only=False)
        h=digest_state(payload['model'])
        assert payload['trainer_state']['completed_epochs']==identity['epoch']==1000
        assert h==digest_state(payload['ema'])==identity['ema_sha256']
        assert payload['run_identity']['config_sha256']==cfg['config_sha256']
        assert cfg['ema_enabled'] is False and cfg['ema_decay'] is None
        assert cfg['external_source']=='MS' and not cfg['internal_noise']
        assert identity['source']==cfg['source_specification']
        same_weights=h==digest_state(old['model'])
        del payload,old
        keys=[(r['scene_id'],r['steps'],r['realization']) for r in data['records']]
        assert len(keys)==len(set(keys))==2560 and set(keys)==expected
        assert len(data['aggregated'])==16 and data['n_scenes']==160 and identity['realizations']==4
        index=dict(zip(keys,data['records']));count=0;max_error=0.
        for n in (2,8):
            for k in range(4):
                files=list((directory/f'test_diagnostics_predictions/N{n}/seed_{k}').glob('*.npz'))
                assert {f.stem for f in files}==scenes
                for f in files:
                    with np.load(f,allow_pickle=False) as z:
                        pred,target,mask=z['prediction'],z['target'],z['valid_mask']
                        assert pred.shape==target.shape==mask.shape==(1,256,256)
                        assert mask.dtype==bool and mask.any() and np.isfinite(pred).all()
                        error=(np.clip(pred,0,1).astype('float64')-target.astype('float64'))[mask]*300
                        rmse=float(np.sqrt(np.mean(error**2)))
                        difference=abs(rmse-index[f.stem,n,k]['db_rmse'])
                        assert difference<1e-5,(v,f,difference)
                        max_error=max(max_error,difference);count+=1
                        th=hashlib.sha256(target.tobytes()+mask.tobytes()).hexdigest()
                        if f.stem in target_hashes:assert target_hashes[f.stem]==th
                        else:target_hashes[f.stem]=th
        history=list(csv.DictReader((directory/'metrics.csv').open()))
        old_history=list(csv.DictReader((ref/'8x8'/v/'metrics.csv').open()))
        assert len(history)==len(old_history)==1000
        same_losses=all(a['train_loss']==b['train_loss'] for a,b in zip(history,old_history))
        old_online=json.loads((directory/'reference_online/test_diagnostics.json').read_text())
        same_test=data['records']==old_online['records'] and data['aggregated']==old_online['aggregated']
        checks[v]=dict(checkpoint_matches=True,online_not_averaged=True,online_sha256=h,
            endpoint_equal_to_old_online=same_weights,all_1000_training_losses_exactly_equal=same_losses,
            test_records_equal_to_old_online=same_test,metric_records=2560,predictions_verified=count,
            max_rmse_recompute_difference=max_error,test_json_sha256=hashlib.sha256((directory/'test_diagnostics.json').read_bytes()).hexdigest())
        print(v,checks[v],flush=True)
    atomic(out/'test_audit.json',dict(status='passed',variants=checks,scenes=160,realizations=4,
        steps=[2,4,8,12],training_seeds=[42],frozen_files_verified=len(receipt),
        inference='Full tests executed by existing suite; checked here without duplicate inference',
        all_saved_test_predictions_verified=5120,legacy_ema_key='Online weights, verified equal to model'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root.resolve())
