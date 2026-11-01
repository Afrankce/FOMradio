"""Isolated 8x8 source-only runner, same two-GPU synchronization as width32."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler,Subset
from data_loaders.multiconfig import multiconfig_collate
from model.conditional_operator_uno import ConditionalOperatorUNO2d, VARIANTS
from model.fno import count_tensor_parameters,count_real_scalar_parameters
from model.matern_source import SOURCES
from training.matern_source import digest_state,evaluate_dataset
from training.conditional_operator_experiment import ConditionalConfig,ConditionalTrainer
from training.complex_grad_scaler import ComplexGradScaler
from training.multiconfig_trainer import seed_everything
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_trainer import preflight_same_frequency,build_same_frequency_checkpoint_identity,infer_manifest_selection,_atomic_json
from training.width32_distributed import source_digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('dataset-root','manifest-path','height-stats-path','run-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',choices=('MS',),default='MS')
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--resume',choices=('auto','none'),default='none')
    parser.add_argument('--smoke-optimizer-steps',type=int)
    parser.add_argument('--stop-after-epoch',type=int)
    parser.add_argument('--evaluate',choices=('val','test'))
    args=parser.parse_args()
    rank,world=int(os.getenv('RANK','0')),int(os.getenv('WORLD_SIZE','1'))
    local=int(os.getenv('LOCAL_RANK','0'));torch.set_num_threads(2);seed_everything(42)
    model=ConditionalOperatorUNO2d(args.variant,operator_width=32)
    assert not any('injection' in k for k in model.state_dict())
    initial_hash=digest_state(model.state_dict())
    common_hash=digest_state({k:v for k,v in model.state_dict().items() if not any(s in k for s in ('channel_attention','spatial_attention','condition_operator'))})
    beam_id,_=infer_manifest_selection(args.manifest_path,'8x8')
    root=args.run_root/'_smoke' if args.smoke_optimizer_steps else args.run_root
    rank_root=root if rank==0 else root/f'rank_{rank}'
    base=SameFrequencyTrainConfig(dataset_root=args.dataset_root,manifest_path=args.manifest_path,
        height_stats_path=args.height_stats_path,run_root=rank_root,array_size='8x8',beam_id=beam_id)
    cfg=ConditionalConfig(base,world,count_tensor_parameters(model),count_real_scalar_parameters(model),
                     source_digest(Path(__file__).parent),source=args.source,condition_variant=args.variant)
    context=preflight_same_frequency(cfg)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    try:
        record={**cfg.to_record(),'initial_model_sha256':initial_hash,'common_model_sha256':common_hash}
        config=rank_root/'config.json'
        if config.exists() and json.loads(config.read_text())!=record:
            raise RuntimeError('Refusing incompatible source configuration')
        if not config.exists(): _atomic_json(config,record)
        last=rank_root/'last.pt'
        if last.exists() and args.resume=='none' and not args.evaluate:
            raise RuntimeError('Refusing to overwrite training')
        model.to(device)
        for value in model.state_dict().values():
            dist.broadcast(torch.view_as_real(value) if value.is_complex() else value,src=0)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
        generator=torch.Generator().manual_seed(42+rank)
        sampler=DistributedSampler(context.train_dataset,num_replicas=world,rank=rank,shuffle=True,seed=42)
        if len(context.train_dataset)%world: raise RuntimeError('Sampler repeats scenes')
        train=DataLoader(context.train_dataset,batch_size=cfg.micro_batch_size,sampler=sampler,generator=generator,
                         num_workers=2,collate_fn=multiconfig_collate,pin_memory=True)
        val=DataLoader(context.val_dataset,batch_size=cfg.eval_batch_size,shuffle=False,num_workers=2 if rank==0 else 0,
                       collate_fn=multiconfig_collate)
        identity=build_same_frequency_checkpoint_identity(cfg,context,model)
        trainer=ConditionalTrainer(cfg,model,train,val,device,generator,identity,scaler=ComplexGradScaler('cuda',enabled=True))
        if rank==0 and not (root/'source_specification.json').exists():
            _atomic_json(root/'source_specification.json',trainer.source_sampler.specification())
        if args.evaluate:
            result=[None]
            if rank==0:
                try:
                    payload=torch.load(root/'last.pt',map_location=device,weights_only=False)
                    if payload['trainer_state']['completed_epochs']!=1000:
                        raise RuntimeError('Final evaluation requires epoch1000')
                    if payload['run_identity']['config_sha256']!=cfg.config_sha256: raise RuntimeError('Checkpoint mismatch')
                    trainer.ema.ema_model.load_state_dict(payload['ema']);del payload
                    dataset=context.val_dataset if args.evaluate=='val' else context.test_dataset
                    evaluate_dataset(trainer.ema.ema_model,trainer.source_sampler,dataset,device,root/f'{args.evaluate}_diagnostics.json',
                        steps=(2,4,8,12,32) if args.evaluate=='val' else (2,4,8,12),time_diagnostics=args.evaluate=='val')
                except Exception as error: result[0]=repr(error)
            dist.broadcast_object_list(result,src=0)
            if result[0]: raise RuntimeError(result[0])
            return
        if args.resume=='auto':
            existence=[None]*world;dist.all_gather_object(existence,last.exists())
            if any(existence) and not all(existence): raise RuntimeError('Incomplete distributed checkpoint')
            if all(existence):
                trainer.resume(last);states=[None]*world
                dist.all_gather_object(states,(trainer.completed_epochs,trainer.optimizer_step))
                marker=json.loads((root/'committed_epoch.json').read_text())
                if len(set(states))!=1 or marker['completed_epochs']!=trainer.completed_epochs:
                    raise RuntimeError('Uncommitted distributed checkpoint')
                if trainer.completed_epochs and trainer.completed_epochs%100==0:
                    # Recover a committed checkpoint whose milestone was interrupted.
                    result=[None]
                    if rank==0:
                        try: trainer.milestone()
                        except Exception as error: result[0]=repr(error)
                    dist.broadcast_object_list(result,src=0)
                    if result[0]: raise RuntimeError(result[0])
        if args.smoke_optimizer_steps:
            result=trainer.run_smoke(args.smoke_optimizer_steps)
            hashes=[digest_state(network.state_dict()) for network in (model,trainer.ema.ema_model)]
            gathered=[None]*world;dist.all_gather_object(gathered,hashes)
            if any(x!=hashes for x in gathered): raise RuntimeError('Rank parameter divergence')
            trainer.resume(Path(result['checkpoint']))
            if [digest_state(network.state_dict()) for network in (model,trainer.ema.ema_model)]!=hashes:
                raise RuntimeError('Checkpoint roundtrip changed parameters')
            result.update(validation=trainer.validate(),rank_checksums_equal=True,initial_model_sha256=initial_hash)
            error=[None]
            if rank==0:
                try:
                    evaluate_dataset(trainer.ema.ema_model,trainer.source_sampler,Subset(context.val_dataset,[0,1]),
                        device,root/'debug_diagnostics.json',steps=(2,8),realizations=1,epoch=0,time_diagnostics=True)
                    _atomic_json(root/'smoke_verified.json',result)
                except Exception as caught: error[0]=repr(caught)
            dist.broadcast_object_list(error,src=0)
            if error[0]: raise RuntimeError(error[0])
        else:
            result=trainer.fit(stop_after_epoch=args.stop_after_epoch)
        if rank==0: print(json.dumps(result),flush=True)
    finally: dist.destroy_process_group()


if __name__=='__main__': main()
