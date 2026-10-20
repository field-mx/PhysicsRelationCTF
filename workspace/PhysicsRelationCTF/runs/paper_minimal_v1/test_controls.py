"""Behavior tests: silent branch deletion, incorrect extra-input axes, lost levels."""
import importlib.util
import sys
from pathlib import Path
import copy
import unittest
import tempfile
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]

def api():
    assert importlib.util.find_spec('controls') is not None, 'controls implementation missing'
    import controls
    return controls

def test_masked_strength_keeps_unablated_parameters_trainable():
    from relation_model import RelationAttention
    from layers.SelfAttention_Family import FullAttention
    for variant,removed in [('no_bias',0),('no_equation',1)]:
        a=RelationAttention(FullAttention(True,attention_dropout=0),.05,.75,.001)
        old=a.strengths()
        api().mask_strength(a,variant)
        values=a.strengths()
        assert values[removed].item()==0
        for j in set(range(3))-{removed}:torch.testing.assert_close(values[j],old[j])
        sum(values).backward()
        gradients=[a.theta_beta.grad,a.theta_gamma.grad,a.theta_rho.grad]
        assert gradients[removed].item()==0
        assert all(gradients[j].abs().item()>0 for j in set(range(3))-{removed})

def test_no_bias_retains_conditional_physical_messages():
    sys.path[:0]=[str(ROOT/'runs/publicdata_exp/optimization_v2'),str(ROOT/'runs/publicdata_exp')]
    from model_v5 import ConditionalRelationAttention
    from layers.SelfAttention_Family import FullAttention
    a=ConditionalRelationAttention(FullAttention(True,attention_dropout=0),.05,.75,.001)
    a.context=dict(qs=torch.tensor([1]),ks=torch.tensor([0]),ei=torch.tensor([0]),
        bias=torch.ones(1,1),physics=torch.ones(1,1,2),residual=torch.ones(1,1,2),
        target_select=torch.tensor([[0.,1.]]),lag_diagnostics={})
    api().mask_strength(a,'no_bias')
    q=torch.zeros(1,2,1,2);v=torch.ones_like(q);mask=torch.zeros(1,1,2,2,dtype=torch.bool)
    actual,_=a(q,q,v,mask)
    assert torch.all(actual[:,1]>1.1), 'No bias must retain nonzero gamma message'
    torch.testing.assert_close(a.context['bias'],torch.ones(1,1))

def test_feature_control_keeps_original_input_and_exposes_constant_level():
    from physics import OriginalCTF,config
    from types import SimpleNamespace
    cfg=config(3,1e-4);cfg.update(d_model=16,n_heads=2,d_ff=32,e_layers=1,dropout=0.)
    base=OriginalCTF(SimpleNamespace(**cfg)).eval()
    def bank(raw):
        eq=torch.stack([raw[...,0]*2,torch.full_like(raw[...,0],15.)],-1)
        valid=torch.ones_like(eq,dtype=torch.bool);valid[:,0,0]=False
        return eq,valid
    model=api().EquationInputCTF(cfg,base,bank,[0,0,0],[1,1,1],
                               [0,10],[2,5]).eval()
    x=torch.randn(2,96,3)
    # Catch: feature channels accidentally instance-centered to zero or wrong axis.
    captured=[]
    handle=model.channel_wise_patch_embedding.register_forward_pre_hook(
        lambda m,args:captured.append(args[0].detach().clone()))
    y=model(x,None,None,None)
    handle.remove()
    assert y.shape==(2,96,3) and torch.isfinite(y).all()
    z=captured[0]
    assert z.shape==(2,7,96)
    torch.testing.assert_close(z[:,4],torch.ones(2,96))
    assert torch.all(z[:,3,0]==0) and torch.all(z[:,5,0]==0)
    torch.testing.assert_close(z[:,3,1:],x[:,1:,0])
    expected=(x-x.mean(1,keepdim=True))/(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()
    torch.testing.assert_close(z[:,:3].transpose(1,2),expected)
    for name,value in base.state_dict().items():
        actual=model.state_dict()[name]
        if actual.shape!=value.shape:actual=actual[:len(value)]
        torch.testing.assert_close(actual,value,atol=0,rtol=0)
    y.square().mean().backward()
    assert model.channel_wise_patch_embedding.embedding_dict['pl24'].weight.grad.abs().sum()>0
    other=api().EquationInputCTF(cfg,base,bank,[0,0,0],[1,1,1],[0,10],[2,5]).eval()
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(other(x,None,None,None),y,atol=0,rtol=0)

def test_feature_standardization_uses_only_valid_values():
    values=torch.tensor([[1.,10.],[3.,20.],[999.,30.]])
    valid=torch.tensor([[1,0],[1,0],[0,0]],dtype=torch.bool)
    mean,scale=api().feature_statistics(values,valid)
    torch.testing.assert_close(mean,torch.tensor([2.,0.]))
    torch.testing.assert_close(scale,torch.tensor([1.,1.]))

def test_experiment_rejects_incomplete_freeze_and_changed_protocol():
    assert (HERE/'supplement.py').exists(),'runner implementation missing'
    import supplement as run
    with tempfile.TemporaryDirectory() as directory:
        tmp_path=Path(directory)
        run.immutable_json(tmp_path/'protocol.json',{'variant':'no_bias'})
        with unittest.TestCase().assertRaises(RuntimeError):
            run.immutable_json(tmp_path/'protocol.json',{'variant':'no_equation'})
        with unittest.TestCase().assertRaises(RuntimeError):
            run.lock_checkpoints([tmp_path/'missing'])

def test_schedule_is_complete_and_outputs_are_isolated():
    import supplement as run
    jobs=run.jobs()
    assert len(jobs)==27
    assert len({tuple(sorted(j.items())) for j in jobs})==27
    for dataset in ('industrial','lbnl','tep'):
        for seed in (42,43,44):
            assert {j['variant'] for j in jobs if j['dataset']==dataset and j['seed']==seed}=={
                'no_bias','no_equation','equation_input'}
    assert all(run.output(j).is_relative_to(HERE) for j in jobs)

def test_fitted_constant_reference_is_not_erased():
    values=torch.tensor([[15.,1.],[15.,3.],[15.,999.]])
    valid=torch.tensor([[1,1],[1,1],[1,0]],dtype=torch.bool)
    mean,scale=api().feature_statistics(values,valid)
    transformed=(values-mean)/scale
    torch.testing.assert_close(transformed[:,0],torch.ones(3))
    torch.testing.assert_close(transformed[:2,1],torch.tensor([-1.,1.]))

def test_ablated_state_cannot_silently_load_as_full_model():
    from relation_model import RelationAttention
    from layers.SelfAttention_Family import FullAttention
    def build():
        model=torch.nn.Module();model.encoder=torch.nn.Module()
        layer=torch.nn.Module();layer.self_attention=torch.nn.Module()
        layer.self_attention.inner_attention=RelationAttention(FullAttention(True),.05,.75,.001)
        model.encoder.layers=torch.nn.ModuleList([layer]);return model
    model=api().ablate(build(),'no_bias')
    with unittest.TestCase().assertRaises(RuntimeError):
        build().load_state_dict(model.state_dict())

def test_worker_has_exclusive_run_lock():
    import supplement as run
    import fcntl
    assert hasattr(run,'exclusive_run'),'Missing per-worker run lock'
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        with run.exclusive_run(out):
            with (out/'worker.lock').open('a') as second:
                with unittest.TestCase().assertRaises(BlockingIOError):
                    fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (out/'worker.lock').open('a') as after:
            fcntl.flock(after,fcntl.LOCK_EX|fcntl.LOCK_NB)

def test_durable_control_identity_and_completion_marker():
    import supplement as run
    assert hasattr(run,'finalize_artifacts'),'Missing durable control labeling'
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);job=dict(dataset='industrial',seed=42,variant='no_bias')
        run.dump(out/'training.json',dict(kind='Physics',mse=1))
        run.dump(out/'config.json',dict(kind='Physics'))
        torch.save(dict(kind='Physics',model={'_paper_control_id':torch.tensor(1)}),out/'best.pth')
        run.finalize_artifacts(out,job)
        assert run.read(out/'training.json')['kind']=='no_bias'
        assert run.read(out/'config.json')['kind']=='no_bias'
        ck=torch.load(out/'best.pth',weights_only=False)
        assert ck['control_variant']=='no_bias' and ck['kind']=='no_bias'
        assert run.read(out/'paper_training.json')['backend_kind']=='Physics'
        run.finalize_artifacts(out,job)  # Idempotent recovery after an interruption.

if __name__=='__main__':
    torch.set_num_threads(2)
    suite=unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in list(globals().items()) if k.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
