"""Isolated component removals and equal-equation-information CTF control."""
import copy
from types import MethodType, SimpleNamespace
import torch
from physics import OriginalCTF


def mask_strength(attention,variant):
    if variant not in ('no_bias','no_equation'):
        raise ValueError(variant)
    if getattr(attention,'ablation_variant',None):
        raise RuntimeError('Do not apply a strength mask twice')
    original=attention.strengths
    removed=0 if variant=='no_bias' else 1
    def strengths(self):
        values=original()
        return tuple(v*0 if i==removed else v for i,v in enumerate(values))
    attention.strengths=MethodType(strengths,attention)
    attention.ablation_variant=variant


def ablate(model,variant):
    model.register_buffer('_paper_control_id',torch.tensor(1 if variant=='no_bias' else 2))
    for layer in model.encoder.layers:
        mask_strength(layer.self_attention.inner_attention,variant)
    return model


def feature_statistics(values,valid):
    if values.shape!=valid.shape or values.ndim!=2:
        raise ValueError('Require point by feature values and validity')
    valid=valid & torch.isfinite(values)
    clean=torch.where(valid,values,torch.zeros_like(values))
    count=valid.sum(0).clamp_min(1)
    mean=clean.sum(0)/count
    centered=torch.where(valid,values-mean,torch.zeros_like(values))
    scale=(centered.square().sum(0)/count).sqrt()
    constant=scale<=1e-8
    # A known, nonzero fixed boundary must not disappear under centering.
    scale=torch.where(constant,mean.abs().clamp_min(1),scale)
    mean=torch.where(constant,torch.zeros_like(mean),mean)
    return mean,scale


class EquationInputCTF(OriginalCTF):
    """Append Train-standardized equations and validity as extra input channels.

    Measured channels keep native instance normalization; extra equation inputs
    bypass instance centering so known absolute references remain observable.
    No graph, relation type, lag matching, confidence gate or value injection.
    Extra forecasts are discarded; no future equation supervision is used.
    """
    def __init__(self,cfg,base,bank,raw_mean,raw_scale,feature_mean,feature_scale):
        cfg=copy.deepcopy(vars(cfg) if hasattr(cfg,'__dict__') else cfg)
        c=cfg['enc_in'];f=len(feature_mean)
        if f<=0 or len(feature_scale)!=f:raise ValueError('Invalid feature scaler')
        cfg['enc_in']=c+2*f
        cfg['patch_lens']=list(cfg['patch_lens'])+[cfg['patch_lens'][0]]*(2*f)
        cfg['sampling_periods']=list(cfg['sampling_periods'])+[cfg['sampling_periods'][0]]*(2*f)
        super().__init__(SimpleNamespace(**cfg))
        self.measured_channels=c;self.equation_bank=bank
        own=self.state_dict()
        for name,value in base.state_dict().items():
            if own[name].shape==value.shape:own[name]=value
            elif name=='channel_wise_patch_embedding.channel_embedding.embedding.weight':own[name][:c]=value
            else:raise RuntimeError(f'Unexpected non-shared backbone parameter: {name}')
        self.load_state_dict(own)
        self.register_buffer('_paper_control_id',torch.tensor(3))
        for name,value in [('raw_mean',raw_mean),('raw_scale',raw_scale),
                           ('feature_mean',feature_mean),('feature_scale',feature_scale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).clone())
        if (self.feature_scale<=0).any():raise ValueError('Nonpositive feature scale')
        self.equation_context=None
        self.channel_wise_patch_embedding.register_forward_pre_hook(self._extra_inputs)
        self.head.register_forward_hook(self._measured_outputs)

    def forecast_multi(self,x_enc,x_mark_enc=None,x_dec=None,x_mark_dec=None):
        if x_enc.shape[1]!=self.seq_len or x_enc.shape[-1]!=self.measured_channels:
            raise ValueError('Unexpected measured input dimensions')
        raw=x_enc*self.raw_scale+self.raw_mean
        values,valid=self.equation_bank(raw)
        valid=valid & torch.isfinite(values)
        values=torch.where(valid,(values-self.feature_mean)/self.feature_scale,torch.zeros_like(values))
        self.equation_context=torch.cat([values,valid.to(values.dtype)],-1)
        return super().forecast_multi(x_enc,x_mark_enc,x_dec,x_mark_dec)

    def _extra_inputs(self,module,args):
        return (torch.cat([args[0],self.equation_context.transpose(1,2)],1),*args[1:])

    def _measured_outputs(self,module,args,output):
        return output[...,:self.measured_channels]
