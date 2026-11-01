"""Matched complete-support regional/Haar analysis at registered N=8 only."""
import numpy as np
from scipy.ndimage import distance_transform_edt
def mean2(a):return (a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])/4

def regional_analysis(stage,variant,epoch,dataset,manifest):
    rows=[];wave=[]
    for file in sorted(stage.glob('N8/seed_*/*.npz')):
        scene=file.stem;sample=np.load(file,allow_pickle=False)
        mask=sample['valid_mask'][0].astype(bool);target=sample['target'][0]*300-300
        raw=sample['prediction'][0];pred=np.clip(raw,0,1)*300-300
        height=np.load(dataset/manifest[scene]['height_path'],allow_pickle=False)
        if height.shape!=mask.shape:raise ValueError('Height/benchmark grid mismatch')
        building=height>0
        dist=distance_transform_edt(~building) if building.any() else np.full(height.shape,np.inf)
        identity=dict(variant=variant,epoch=epoch,scene=scene,steps=int(file.parent.parent.name[1:]),realization=int(file.parent.name[5:]))
        for name,region in [('all',mask),('near_building_le8px',mask&(dist<=8)),('far_building_gt16px',mask&(dist>16))]:
            error=(pred-target)[region]
            rows.append(dict(**identity,region=name,count=int(region.sum()),
                             rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                             bias=float(np.mean(error)) if len(error) else None))
        p,y,valid=mean2(pred),mean2(target),mean2(mask.astype(float))==1
        for level in (1,2):
            support=mean2(valid.astype(float))==1
            def haar(a):
                aa,bb,cc,dd=a[::2,::2],a[::2,1::2],a[1::2,::2],a[1::2,1::2]
                return (aa+bb+cc+dd)/2,[(aa-bb+cc-dd)/2,(aa+bb-cc-dd)/2,(aa-bb-cc+dd)/2]
            p,pd=haar(p);y,yd=haar(y)
            for orientation,a,b in zip(('column_difference','row_difference','diagonal_difference'),pd,yd):
                av,bv=a[support],b[support];count=len(av)
                corr=float(np.corrcoef(av,bv)[0,1]) if count>1 and av.std()>0 and bv.std()>0 else None
                wave.append(dict(**identity,level=level,orientation=orientation,count=count,
                    mse=float(np.mean((av-bv)**2)) if count else None,
                    pred_coefficient_mean_square=float(np.mean(av**2)) if count else None,
                    target_coefficient_mean_square=float(np.mean(bv**2)) if count else None,
                    signed_coefficient_correlation=corr))
            valid=support
    return rows,wave
