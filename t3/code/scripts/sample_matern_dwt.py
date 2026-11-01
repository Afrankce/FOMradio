"""Sample the exact current source implementation; no training/data mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.matern_source import SourceSampler,source_spectra,SOURCES


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--draws',type=int,default=512);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'samples.npy').exists():raise RuntimeError('Use a fresh output; no overwrite')
    torch.set_num_threads(2)
    samplers={s:SourceSampler(s) for s in SOURCES}
    specs={s:v.specification() for s,v in samplers.items()}
    previous=Path(__file__).resolve().parents[1]/'runs/no_ema_ms_8x8_20260910T131052Z/8x8/CASA/source_specification.json'
    assert specs['MS']==json.loads(previous.read_text())
    field=np.lib.format.open_memmap(out/'samples.npy',mode='w+',dtype=np.float32,shape=(len(SOURCES),a.draws,256,256))
    for start in range(0,a.draws,8):
        metadata=[dict(scene_id=f'dwt_probe_{i:04d}',steering_deg=0.) for i in range(start,min(start+8,a.draws))]
        white=samplers['W'].base_noise(metadata,a.device,'dwt_probe_v1')
        for j,s in enumerate(SOURCES):field[j,start:start+len(metadata)]=samplers[s].transform(white)[:,0].cpu().numpy()
        if start%64==0:print('sampled',start+len(metadata),'/',a.draws,flush=True)
    field.flush();del field
    extent,spectra=source_spectra()
    np.save(out/'embedding_spectra.npy',spectra)
    record=dict(status='complete',specifications=specs,sources=list(SOURCES),draws=a.draws,
        device=a.device,namespace='dwt_probe_v1',scene_ids='dwt_probe_0000 through dwt_probe_0511',
        paired='All source variants transform the same real white draw; independent across draw IDs',
        target_data_used=False,per_sample_normalization=False,
        sampler_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'model/matern_source.py').read_bytes()).hexdigest(),
        samples_sha256=hashlib.file_digest((out/'samples.npy').open('rb'),'sha256').hexdigest(),
        spectra_sha256=hashlib.file_digest((out/'embedding_spectra.npy').open('rb'),'sha256').hexdigest())
    (out/'sampling.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
