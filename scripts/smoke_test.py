"""Synthetic forwards and exact interface-off comparison; no data or training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from model_factory import build_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['industrial','lbnl','tep'], required=True)
    parser.add_argument('--ctf-root')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    base, physics, spec = build_pair(args.dataset, ctf_root=args.ctf_root)
    base.eval().to(args.device)
    physics.eval().to(args.device)
    cfg = spec['config']
    x = torch.randn(2, cfg['seq_len'], cfg['enc_in'], device=args.device)
    expected = (2, cfg['pred_len'], cfg['enc_in'])
    with torch.no_grad():
        enhanced = physics(x,None,None,None)
        assert enhanced.shape == expected and torch.isfinite(enhanced).all()
        physics.set_strengths(0.,0.,0.)
        plain = base(x,None,None,None)
        disabled = physics(x,None,None,None)
    error = (plain-disabled).abs().max().item()
    torch.testing.assert_close(plain, disabled, rtol=1e-6, atol=1e-6)
    print(json.dumps(dict(dataset=args.dataset, input_shape=list(x.shape),
          output_shape=list(enhanced.shape), enabled_finite=True,
          interface_off_max_abs_difference=error,
          ctf_parameters=sum(p.numel() for p in base.parameters()),
          prctf_parameters=sum(p.numel() for p in physics.parameters()),
          input_kind='synthetic; not real observations', training=False)))


if __name__ == '__main__':
    main()
