"""Freeze training-only beam gradient statistic and codebook audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from array_physics import codebook
from dataset import QueryLoader
from beam_matern_source import spatial_gradient


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_step4_codebook(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_extended_codebook')
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--step4',type=Path,required=True)
    p.add_argument('--beam-calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cal=json.loads(a.beam_calibration.read_text())
    if not cal['passed']:raise RuntimeError('published-beam calibration failed')
    records=json.loads(a.manifest.read_text());queries=[];seen=set()
    for rec in records:
        q=rec['query']
        if q['beam'] not in seen:queries.append(q);seen.add(q['beam'])
    if len(queries)!=3:raise RuntimeError('expected exactly three training beams')
    loader=QueryLoader(a.root);_,b=loader.inputs(queries,'cpu','C')
    gradients=spatial_gradient(b).numpy();positive=gradients[np.isfinite(gradients)&(gradients>0)]
    if not positive.size:raise ValueError('no positive training-beam gradients')
    scale=max(float(np.percentile(positive,95)),1e-6)
    rt=extract_step4_codebook(a.step4)
    cases=[]
    for n in (8,32):
        for angle in (-14,0,14):
            step4=rt(n,n,angle,0,n,n,'rect').reshape(n,n)
            step6=codebook(n,angle).reshape(n,n)
            err=float(np.max(np.abs(step4.T-step6.conj())))
            cases.append({'n':n,'angle':angle,'conjugate_transpose_max_abs':err})
    if max(x['conjugate_transpose_max_abs'] for x in cases)>1e-10:raise RuntimeError('step4/step6 codebook transform mismatch')
    # Report the inconsistency in settings: the public codebook with subarray=1
    # is angle invariant, whereas the published maps match a full-array codebook.
    one_minus=rt(8,8,-14,0,1,1,'rect');one_plus=rt(8,8,14,0,1,1,'rect')
    report={'training_beam_gradient_p95':scale,'positive_gradient_count':int(positive.size),
            'training_beams':{q['beam']:sha(a.root/q['beam']) for q in queries},
            'step4_sha256':sha(a.step4),'algebraic_codebook_cases':cases,
            'subarray_1_angle_difference':float(np.max(np.abs(one_minus-one_plus))),
            'metadata_conflict':'settings report 1x1 subarrays; stored beam maps match full-array Step6 codebook',
            'limitation':'Power-only archived maps do not independently verify complex RT channel phases or axis ordering.',
            'basis_convention':'Plan plus-sign basis is the reverse index permutation of Step6 minus-sign basis for the symmetric four directions.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
