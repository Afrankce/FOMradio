"""Freeze disjoint scene/configuration lists without auxiliary radio inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ANGLES = (-28,-21,-14,-7,0,7,14,21)
SEEN = (-14,0,14)


def config_for(n):
    if n not in (8,32): raise ValueError('only archived 8x8 and 32x32 arrays')
    return f'freq_6.7GHz_{n*n}TR_{8 if n==8 else 64}beams_pattern_tr38901'


def beam_index(n, angle):
    if n not in (8,32) or angle not in ANGLES: raise ValueError('unavailable target query')
    return (angle+28)//7 if n==8 else angle+32


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_record(root, scene, array, angle, split, group):
    beam=beam_index(array,angle); cfg=config_for(array)
    h=f'height_maps/{scene}/{scene}_height_matrix.npy'
    b=f'beam_maps/{cfg}/u0/beam_{beam:02d}_angle_{float(angle):.1f}_matrix.npy'
    y=f'radiomaps/{cfg}_beam{beam:02d}/{scene}_labeled_radiomap.npy'
    for p in (h,b,y):
        if not (root/p).is_file(): raise FileNotFoundError(root/p)
    return {'query': {'scene':scene, 'array':array, 'angle':angle,
                      'frequency_hz':6700000000, 'height':h, 'beam':b,
                      'beam_index':beam, 'split':split, 'group':group},
            'supervision': {'target': y}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True); p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    split=json.loads(a.split.read_text()); prior=json.loads(a.prior.read_text())
    sizes={'train':560,'val':80,'test':160}
    sets={s:set(split[s]) for s in sizes}
    for s,n in sizes.items():
        assert len(split[s])==len(sets[s])==n and sets[s]==set(prior[s]), s
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    datasets={}
    for s in ('train','val'):
        datasets[s]=[make_record(a.dataset,scene,8,angle,s,'seen') for scene in split[s] for angle in SEEN]
    groups={'seen_config':(8,SEEN),'array':(32,SEEN),'angle_interpolation':(8,(-7,7)),
            'angle_extrapolation':(8,(-28,-21,21)),'joint':(32,(-28,-21,-7,7,21))}
    for group,(array,angles) in groups.items():
        datasets[group]=[make_record(a.dataset,scene,array,angle,'test',group) for scene in split['test'] for angle in angles]
    assert {k:len(v) for k,v in datasets.items()} == {'train':1680,'val':240,'seen_config':480,
                 'array':480,'angle_interpolation':320,'angle_extrapolation':480,'joint':800}
    a.output.mkdir(parents=True,exist_ok=True)
    files={}
    for k,v in datasets.items():
        path=a.output/f'{k}.json'; path.write_text(json.dumps(v,indent=2),encoding='utf-8');files[k]=sha(path)
    beams=sorted({x['query']['beam'] for rows in datasets.values() for x in rows})
    summary={'dataset_root':str(a.dataset),'split_sha256':sha(a.split),'prior_scene_match':True,
             'counts':{k:len(v) for k,v in datasets.items()},'manifests_sha256':files,
             'beam_sha256':{p:sha(a.dataset/p) for p in beams},'auxiliary_map_inputs':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
