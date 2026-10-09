from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from daynight_diffvpr.diffusion_common import (pil_edge, edge_iou, predict_x0,
    differentiable_vpr_input, structure_loss, illumination_loss)
from daynight_diffvpr.stage2_generate import load_day_rows,build_training_triplets,run
from daynight_diffvpr.stage3_joint import (PairedDayNightTriplets, generator_step,
    train_vpr_step,place_objective)
from daynight_diffvpr.models import DescriptorAdapter, TinyCNN


def _img(path, tone):
    Image.fromarray(np.full((32,32,3),tone,dtype=np.uint8)).save(path)


def _csv(path,fields,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_stage2_only_training_split(tmp_path):
    _img(tmp_path/'a.png',120)
    path=tmp_path/'day.csv'
    _csv(path,['id','path','place_id','split'],[dict(id='x',path='a.png',place_id='location1',split='test')])
    with pytest.raises(ValueError,match='Refusing'):
        load_day_rows(path,tmp_path)
    _csv(path,['id','path','place_id','split'],[dict(id='x',path='a.png',place_id='location1',split='train')])
    rows=load_day_rows(path,tmp_path)
    assert rows[0]['place_id']=='location1'


def test_stage2_generate_integration_without_big_weights(tmp_path,monkeypatch):
    _img(tmp_path/'a.png',100)
    _img(tmp_path/'b.png',140)
    path=tmp_path/'day.csv'
    _csv(path,['id','path','place_id','split'],[
        dict(id='a',path='a.png',place_id='p1',split='train'),
        dict(id='b',path='b.png',place_id='p2',split='train')])
    class FakeResult:
        def __init__(self,image):
            self.images=[image]
    class FakePipe:
        def __call__(self,**kwargs):
            return FakeResult(kwargs['image'].point(lambda p:int(p*.25)))
    monkeypatch.setattr('daynight_diffvpr.stage2_generate.load_pipeline',lambda *args:FakePipe())
    config={'train_data':{'image_root':str(tmp_path),'day_csv':str(path)},
        'generation':{'resolution':128,'seed':7,'max_images':2,'min_edge_iou':0}}
    result=run(config,tmp_path/'output',device='cpu')
    assert result['triplets']==2
    assert (tmp_path/'output'/'images'/'000000.png').is_file()
    assert (tmp_path/'output'/'triplets.csv').is_file()
    assert (tmp_path/'output'/'generated.csv').is_file()


def test_stage2_triplets_have_different_negative_places(tmp_path):
    rec=[{'place_id':'p1','source_path':'/abs/day1.png','generated_path':'/abs/gen1.png'},
         {'place_id':'p2','source_path':'/abs/day2.png','generated_path':'/abs/gen2.png'}]
    assert build_training_triplets(rec,tmp_path/'trip.csv') == 2
    with (tmp_path/'trip.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert rows[0]['positive'] != rows[0]['negative']


def test_image_gradients_propagate_through_x0_vpr():
    noisy=torch.randn(1,4,8,8)
    eps=nn.Parameter(torch.randn(1,4,8,8))
    x0=predict_x0(noisy,eps,torch.tensor([0.6]))
    image=nn.Conv2d(4,3,1)(x0).sigmoid()
    trans=differentiable_vpr_input(image,{'data':{'image_size':[16,16],'normalize':'imagenet'}})
    tiny=DescriptorAdapter(TinyCNN(dim=8)).eval()
    tiny.requires_grad_(False)
    loss=tiny(trans).square().sum()
    loss.backward()
    assert eps.grad is not None and eps.grad.abs().sum() > 0
    assert all(p.grad is None for p in tiny.parameters())


def test_structure_gradient_and_night_statistics():
    day=torch.rand(1,3,16,16)
    night=torch.rand(1,3,16,16)
    image=nn.Parameter(torch.rand(1,3,16,16))
    val=structure_loss(day,image)+illumination_loss(image,night)
    val.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_paired_day_night_dataset_checks_negative_label(tmp_path):
    for name in ('day','night','neg'):
        _img(tmp_path/(name+'.png'),100)
    csvpath=tmp_path/'pairs.csv'
    fields=['day','night','negative','place_id','negative_place_id','split']
    good={'day':'day.png','night':'night.png','negative':'neg.png','place_id':'p1','negative_place_id':'p2','split':'train'}
    _csv(csvpath,fields,[good])
    ds=PairedDayNightTriplets(csvpath,tmp_path,128)
    assert ds[0][0].shape==(3,128,128)
    with pytest.raises(ValueError,match='Negative'):
        _csv(csvpath,fields,[{**good,'negative_place_id':'p1'}])
        PairedDayNightTriplets(csvpath,tmp_path,128)


class FakeDiffusion:
    """Lightweight proxy for training-loop gradient integration testing."""
    def __init__(self):
        self.lora_like=nn.Conv2d(3,3,1)
    @property
    def trainable_params(self):
        return list(self.lora_like.parameters())
    def diffusion_loss(self,real_night,control):
        return (self.lora_like(real_night)-real_night).square().mean()
    def day_to_night_proxy(self,day,control,proxy_t):
        return self.lora_like(day).sigmoid()


def test_joint_generator_and_vpr_steps_optimize_both_models():
    torch.manual_seed(1)
    config={
        'data':{'image_size':[16,16], 'normalize':'none'},
        'joint':{'proxy_timestep':300,'lambda_diff':1,'lambda_place':1,
                 'lambda_struct':.03,'lambda_illum':.1,'freeze_vpr_modules':[],
                 'margin':.2,'grad_clip':1},
    }
    diff=FakeDiffusion()
    vpr=DescriptorAdapter(TinyCNN(dim=8))
    optim_g=torch.optim.SGD(diff.trainable_params,lr=.01)
    optim_v=torch.optim.SGD(vpr.parameters(),lr=.01)
    day=torch.rand(1,3,16,16)
    night=torch.rand(1,3,16,16)
    neg=torch.rand(1,3,16,16)
    initial_g=diff.lora_like.weight.detach().clone()
    initial_v=vpr.backbone.features[0].weight.detach().clone()
    generated, logs=generator_step(diff,vpr,optim_g,config,day,night,neg,torch.device('cpu'))
    assert set(logs)=={'diff','place','struct','illum','generator'}
    assert not torch.equal(initial_g,diff.lora_like.weight)
    loss_v=train_vpr_step(vpr,optim_v,config,day,night,neg,generated,torch.device('cpu'))
    assert torch.isfinite(loss_v)
    assert not torch.equal(initial_v,vpr.backbone.features[0].weight)


def test_stage3_place_loss_penalizes_wrong_place():
    day=F.normalize(torch.tensor([[1.,0.,0.]]),dim=1)
    neg=F.normalize(torch.tensor([[0.,1.,0.]]),dim=1)
    assert place_objective(day,day,neg) < place_objective(day,neg,neg)
