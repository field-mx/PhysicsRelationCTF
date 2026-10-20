"""Public-dataset adapters. Existing project and released source are read-only."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'runs/physics_ai_comparison_v1'
sys.path.insert(0, str(LEGACY))
import comparison_models as legacy


def forecast_loss(prediction, truth, targets):
    return (prediction[..., targets] - truth[..., targets]).square().mean()


def augmented_adjacency(graph):
    c = len(graph['channels']); f = len(graph['features'])
    a = torch.eye(c + f)
    for edge in graph['edges']:
        a[edge['target'], edge['source']] = 1.
        fi = edge['feature']
        if fi is None:
            continue
        a[edge['target'], c + fi] = 1.
        feature = graph['features'][fi]
        dependencies = list(feature.get('inputs', []))
        for key in ('flow_index', 'reference_index'):
            if feature.get(key) is not None:
                dependencies.append(feature[key])
        for source in dependencies:
            a[c + fi, source] = 1.
    return a


def route_tolerance(raw, bank, valid, segments, feature, target, lag, length, gain, scale):
    errors = []
    for start, stop in segments:
        t = np.arange(start + length + lag, stop)
        mask = valid[t - lag, feature] & valid[t - length - lag, feature]
        t = t[mask]
        if len(t):
            error = (raw[t, target] - raw[t - length, target] -
                     gain * (bank[t - lag, feature] - bank[t - length - lag, feature])) / scale
            errors.append(np.abs(error[np.isfinite(error)]))
    error = np.concatenate(errors) if errors else np.empty(0)
    return (max(float(np.quantile(error, .90)), .02), len(error)) if len(error) else (.02, 0)


class ShortMTGNN(legacy.MTGNNForecast):
    """Same official MTGNN, RF19 for 20 real observations, no input padding."""
    def __init__(self, cfg, options, seed):
        nn.Module.__init__(self)
        layers = legacy.module_file('_comparison_mtgnn_layer', LEGACY / 'vendor/MTGNN/layer.py')
        old = sys.modules.get('layer'); sys.modules['layer'] = layers
        try:
            network = legacy.module_file('_comparison_mtgnn_net', LEGACY / 'vendor/MTGNN/net.py')
        finally:
            if old is None: sys.modules.pop('layer', None)
            else: sys.modules['layer'] = old
        self.seed = seed; self.seq_len = cfg['seq_len']; self.channels = cfg['enc_in']
        self.network = network.gtnet(
            gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=self.channels,
            device=torch.device('cpu'), dropout=options['dropout'],
            subgraph_size=min(options['neighbors'], self.channels), node_dim=40,
            dilation_exponential=1, conv_channels=32, residual_channels=32,
            skip_channels=64, end_channels=128, seq_length=self.seq_len,
            in_dim=1, out_dim=cfg['pred_len'], layers=3)
        if self.network.receptive_field > self.seq_len:
            raise ValueError('MTGNN requires more real observations than provided')


class PublicPhyGNN(nn.Module):
    """Released G3CN plus explicit equation nodes and forecasting compensation.

    This is a forecasting adaptation, not a replication of the spacecraft SPS
    reconstruction task. Only the same observed equations/graph are provided.
    """
    def __init__(self, cfg, data, options):
        super().__init__()
        spatial = legacy.module_file('_comparison_phygnn_spatial',
                                     LEGACY / 'vendor/PhyGNN/PhyGNN/spatial_block.py')
        self.c = cfg['enc_in']; self.length = cfg['seq_len']; self.bank = data['bank']
        self.register_buffer('adjacency', augmented_adjacency(data['graph']))
        for key in ('raw_mean', 'raw_scale'):
            self.register_buffer(key, torch.tensor(data[key], dtype=torch.float32))
        self.register_buffer('eq_mean', torch.tensor(data['feature_scaler']['mean'], dtype=torch.float32))
        self.register_buffer('eq_scale', torch.tensor(data['feature_scaler']['scale'], dtype=torch.float32))
        self.graph = spatial.CMTS_GCN(options['graph_heads'], self.adjacency.shape[0], CMTS_GCN_residual=.1)
        hidden = options['temporal_hidden']
        self.correction = nn.Sequential(nn.Linear(2*self.length, hidden), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(hidden, cfg['pred_len']))
        nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)
        route = torch.zeros(self.c, len(data['graph']['features'])); counts = torch.zeros(self.c)
        seen = set()
        for edge, gain in zip(data['graph']['edges'], data['scalers']['gains']):
            fi, target = edge['feature'], edge['target']
            if fi is None or (fi, target) in seen:
                continue
            seen.add((fi, target)); route[target, fi] = gain / self.raw_scale[target]
            counts[target] += 1
        route /= counts.clamp_min(1)[:, None]
        self.register_buffer('physical_route', route)
        self.physical_forecast = nn.Linear(self.length, cfg['pred_len'], bias=False)
        nn.init.zeros_(self.physical_forecast.weight)
        with torch.no_grad(): self.physical_forecast.weight[:, -1] = 1.

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        if x.shape[1:] != (self.length, self.c):
            raise ValueError('Unexpected history dimensions')
        raw = x*self.raw_scale + self.raw_mean
        equation, valid = self.bank(raw)
        eq_input = torch.where(valid, (equation-self.eq_mean)/self.eq_scale, torch.zeros_like(equation))
        graph_history = self.graph(torch.cat((x, eq_input), -1).transpose(1, 2), self.adjacency)[:, :self.c]
        mean = x.mean(1, keepdim=True).detach()
        scale = (x.var(1, keepdim=True, unbiased=False)+1e-5).sqrt()
        own = ((x-mean)/scale).transpose(1, 2)
        variations = torch.where(valid & valid[:, -1:], equation-equation[:, -1:], torch.zeros_like(equation))
        mapped = torch.einsum('btf,cf->btc', variations, self.physical_route)
        reference = self.physical_forecast((x[:, -1:]+mapped).transpose(1, 2)).transpose(1, 2)
        residual = self.correction(torch.cat((own, graph_history), -1)).transpose(1, 2)
        return reference + scale*residual


class PartialPhysicsLoss(nn.Module):
    """Train-calibrated tolerance-aware proxies, not a closed-system PINN."""
    def __init__(self, data):
        super().__init__()
        self.bank = data['bank']; self.length = data['config']['seq_len']
        self.channels = data['graph']['channels']
        for key in ('raw_mean', 'raw_scale'):
            self.register_buffer(key, torch.tensor(data[key], dtype=torch.float32))
        # Control/status/reference channels are historical inputs only. Future
        # unscored controls are persisted, not supplied from future observations.
        keep = torch.zeros(len(self.channels), dtype=torch.bool); keep[data['targets']] = True
        self.register_buffer('forecasted', keep)
        raw = data['raw_train']
        with torch.no_grad():
            bank, valid = self.bank(torch.as_tensor(raw, dtype=torch.float64))
        bank, valid = bank.numpy(), valid.numpy()
        seen = set(); routes = []
        for edge, gain in zip(data['graph']['edges'], data['scalers']['gains']):
            fi, target = edge['feature'], edge['target']
            if fi is None or target not in data['targets'] or (fi, target) in seen:
                continue
            seen.add((fi, target)); lags = sorted(edge['lags']); lag = int(lags[len(lags)//2])
            if lag >= self.length: raise ValueError('Physics lag outside real history')
            tolerance, count = route_tolerance(raw, bank, valid, data['segments'], fi, target,
                                              lag, self.length, float(gain), data['raw_scale'][target])
            if count:
                routes.append(dict(feature=fi, feature_name=data['graph']['features'][fi]['name'],
                    target=target, target_name=self.channels[target], lag=lag,
                    gain=float(gain), tolerance=tolerance, train_pairs=count))
        self.routes = routes
        for name, dtype in [('feature', torch.long), ('target', torch.long), ('lag', torch.long),
                            ('gain', torch.float32), ('tolerance', torch.float32)]:
            self.register_buffer(name, torch.tensor([r[name] for r in routes], dtype=dtype))

    def forward(self, history, prediction):
        if not self.routes: return prediction.sum()*0
        future = torch.where(self.forecasted[None, None], prediction,
                             history[:, -1:].expand_as(prediction))
        combined = torch.cat((history, future), 1)*self.raw_scale + self.raw_mean
        bank, valid = self.bank(combined)
        t = torch.arange(prediction.shape[1], device=prediction.device)
        index = self.length+t[:, None]-self.lag[None]
        initial = self.length-1-self.lag
        source_delta = bank[:, index, self.feature[None]] - bank[:, initial, self.feature][:, None]
        expected = source_delta*self.gain[None, None]/self.raw_scale[self.target][None, None]
        observed_delta = prediction[:, :, self.target]-history[:, -1:, self.target]
        active = valid[:, index, self.feature[None]] & valid[:, initial, self.feature][:, None]
        excess = F.relu((observed_delta-expected).abs()-self.tolerance[None, None])
        return (excess.square()*active).sum()/active.sum().clamp_min(1)


def make_model(kind, cfg, data, seed, options):
    legacy.seed_all(seed)
    if kind == 'MTGNN':
        cls = ShortMTGNN if cfg['seq_len'] < 43 else legacy.MTGNNForecast
        return cls(cfg, options, seed)
    if kind == 'PatchTST':
        return legacy.PatchTSTForecast(cfg, options)
    if kind == 'PhyGNN':
        return PublicPhyGNN(cfg, data, options)
    if kind == 'PG-Reg-CTF':
        controls = legacy.module_file('_comparison_equation_input', ROOT/'runs/paper_minimal_v1/controls.py')
        base = legacy.OriginalCTF(SimpleNamespace(**copy.deepcopy(cfg)))
        fs = data['feature_scaler']
        return controls.EquationInputCTF(SimpleNamespace(**copy.deepcopy(cfg)), base, data['bank'],
                                       data['raw_mean'], data['raw_scale'], fs['mean'], fs['scale'])
    raise ValueError(kind)
