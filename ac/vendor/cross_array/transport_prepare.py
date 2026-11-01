"""Create an isolated T3 run; never mutate its completed T2 reference."""
import argparse
from pathlib import Path
import shutil
import time
import torch
from beam_boundary.common import atomic, read, sha, verify, LABELS, state_sha
from .model import SourceObservationUNO


def verify_reference(root):
    audit=read(root/'reference_audit.json');parent=Path(audit['parent'])
    for name,digest in audit['sha256'].items():
        if sha(parent/name)!=digest: raise RuntimeError(f'T2 reference changed: {name}')


def prepare(root,parent,project):
    if root.exists(): raise RuntimeError('Refusing to overwrite an existing run')
    verify(parent)
    assert read(parent/'suite.json')['status']=='complete'
    assert read(parent/'progression_gate.json')['decision']=='supported'
    old=read(parent/'config.json');assert old['experiment_kind']=='cross_array_observation'
    assert old['world_size']==2 and old['seed']==42 and old['total_steps']==5000
    assert old['source_observation_angle']==0. and old['ema'] is False
    reference_paths=['config.json','snapshot.json','input_audit.json','progression_gate.json','jobs/T2/committed.json']
    commit=read(parent/'jobs/T2/committed.json');reference_paths.append('jobs/T2/'+commit['checkpoint'])
    for split,n in [('val',80),('test',160)]:
        folder=parent/'jobs/T2/evaluation/step_005000'/split
        done=read(folder/'complete.json')
        assert done['step']==5000 and done['draws']==4 and done['nfe']==8 and done['scenes']==n
        assert done['checkpoint_sha256']==commit['sha256']
        reference_paths += [str((folder/f).relative_to(parent)) for f in ('summary.json','complete.json')]
    for step in (1000,2500):
        reference_paths += [f'jobs/T2/evaluation/step_{step:06d}/val/{f}' for f in ('summary.json','complete.json')]
    root.mkdir(parents=True)
    (root/'data').symlink_to(parent/'data',target_is_directory=True)
    shutil.copy2(parent/'input_audit.json',root/'input_audit.json')
    cfg=dict(old);cfg.update(schema=4,experiment_kind='cross_array_transport',arms=['T3'],new_training_arms=['T3'],
        display_arms=['T2','T3'],reference_arms=['T2'],baseline_arm='T2',candidate_arm='T3',
        labels=LABELS,parent_run=str(parent),allowed_gpu_ids=list(range(8)),comparison_pairs=[['T3','T2']],
        source_mean_rule='fixed_real_y8_0deg_masked_zero',source_noise_scale=1.,
        physical_claim='Conditional source translation in normalized scalar dB-map coordinates, not RF time',created=time.time())
    atomic(root/'config.json',cfg)
    atomic(root/'reference_audit.json',dict(parent=str(parent),sha256={f:sha(parent/f) for f in reference_paths},
        policy='Read-only completed T2; no retraining; identical data and training budget'))
    ref=root/'jobs/T2';ref.mkdir(parents=True)
    for name in ('history.json','committed.json','evaluation'):
        (ref/name).symlink_to(parent/'jobs/T2'/name,target_is_directory=name=='evaluation')
    atomic(ref/'job.json',dict(arm='T2',phase='reference_only_completed',step=5000,updated=time.time(),parent=str(parent)))
    atomic(ref/'progress.json',dict(arm='T2',phase='reference_only_completed',step=5000,updated=time.time()))
    atomic(root/'jobs/T3/job.json',dict(arm='T3',phase='queued',updated=time.time()))
    torch.set_num_threads(2)
    torch.manual_seed(42);a=SourceObservationUNO('T2')
    torch.manual_seed(42);b=SourceObservationUNO('T3')
    assert state_sha(a.state_dict())==state_sha(b.state_dict())
    atomic(root/'architecture_audit.json',dict(passed=True,initial_sha256=state_sha(a.state_dict()),
        parameter_count=sum(p.numel() for p in a.parameters()),identical_initial_state=True,changed_factor='FM source mean only'))
    shutil.copytree(parent/'code',root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('beam_boundary','cross_array'):
        shutil.copytree(project/name,root/'code'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('test_cross_array.py','test_cross_array_transport.py'):
        shutil.copy2(project/'tests'/name,root/'code'/name)
    shutil.copy2(project/'docs/CROSS_ARRAY_TRANSPORT.md',root/'PROTOCOL.md')
    atomic(root/'snapshot.json',dict(code_sha256={str(p.relative_to(root/'code')):sha(p)
        for p in sorted((root/'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        parent_snapshot_sha256=sha(parent/'snapshot.json'),protocol_sha256=sha(root/'PROTOCOL.md')))
    atomic(root/'suite.json',dict(status='prepared',updated=time.time()))
    print(root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    prepare(a.root.resolve(),a.parent.resolve(),a.project.resolve())
