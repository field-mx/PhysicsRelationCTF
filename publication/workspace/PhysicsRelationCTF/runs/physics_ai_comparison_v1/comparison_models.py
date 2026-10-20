"""Literature-backed adapters; existing CTF/physics sources remain untouched."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from physics import FEATURES, OriginalCTF, equation_features, seed_all


DEPENDENCIES = {
    'power_W': ('SCRDLFK', 'SCRDYFK'),
    'H2_per_cell_mol_s': ('SCRDLFK',),
    'O2_per_cell_mol_s': ('SCRDLFK',),
    'cooling_delta_K': ('TT010002', 'TT010001'),
    'O2_lye_delta_K': ('TT010401', 'TT010301'),
    'H2_lye_delta_K': ('TT010403', 'TT010301'),
    'O2_heat_proxy': ('FT010401', 'TT010401', 'TT010301'),
    'H2_heat_proxy': ('FT010401', 'TT010403', 'TT010301'),
    'O2_valve_dp_MPa': ('PT010303', 'PT010501'),
    'separator_dp_MPa': ('PT010302', 'PT010301'),
}


def module_file(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sensor_adjacency(graph):
    """Query/receiver rows, source columns. No graph powers or reverse edges."""
    c = len(graph['channels'])
    a = torch.zeros(c, c)
    for edge in graph['edges']:
        a[edge['target'], edge['source']] = 1.
    return a


def augmented_adjacency(graph):
    """Add only explicit equation dependency/target routes to the same sensor graph."""
    channels = graph['channels']; c = len(channels)
    a = torch.zeros(c + len(FEATURES), c + len(FEATURES))
    a[:c, :c] = sensor_adjacency(graph)
    ix = {name: i for i, name in enumerate(channels)}
    for edge in graph['edges']:
        feature = edge['feature']
        if feature is None:
            continue
        node = c + FEATURES.index(feature)
        a[edge['target'], node] = 1.
        for name in DEPENDENCIES[feature]:
            if name in ix:
                a[node, ix[name]] = 1.
    return a


class MTGNNForecast(nn.Module):
    """Official mix-hop/dilated-inception network with history-only instance scaling."""
    def __init__(self, cfg, options, seed):
        super().__init__()
        folder = HERE / 'vendor/MTGNN'
        layers = module_file('_comparison_mtgnn_layer', folder / 'layer.py')
        old = sys.modules.get('layer')
        sys.modules['layer'] = layers
        try:
            network = module_file('_comparison_mtgnn_net', folder / 'net.py')
        finally:
            if old is None:
                sys.modules.pop('layer', None)
            else:
                sys.modules['layer'] = old
        self.seed = seed
        self.seq_len = cfg['seq_len']; self.channels = cfg['enc_in']
        self.network = network.gtnet(
            gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=self.channels,
            device=torch.device('cpu'), dropout=options['dropout'],
            subgraph_size=min(options['neighbors'], self.channels), node_dim=40,
            dilation_exponential=2, conv_channels=32, residual_channels=32,
            skip_channels=64, end_channels=128, seq_length=self.seq_len,
            in_dim=1, out_dim=cfg['pred_len'], layers=3)
        if self.network.receptive_field > self.seq_len:
            raise ValueError('MTGNN would pad extra history; select a compatible receptive field')

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        if x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError('Expected historical input only')
        mean = x.mean(1, keepdim=True).detach()
        scale = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        normalized = (x-mean) / scale
        self.network.idx = torch.arange(self.channels, device=x.device)
        self.network.gc.device = x.device
        # Upstream graph_constructor adds random tie-breaking noise even in eval.
        # Make evaluation repeatable, preserving the upstream training behavior.
        if self.training:
            p = self.network(normalized.transpose(1, 2)[:, None])
        else:
            devices = [x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                p = self.network(normalized.transpose(1, 2)[:, None])
        return p[..., 0] * scale + mean


class PatchTSTForecast(nn.Module):
    """Official supervised PatchTST; padding_patch=None means every patch is real."""
    def __init__(self, cfg, options):
        super().__init__()
        folder = HERE / 'vendor/PatchTST/PatchTST_supervised/layers'
        helpers = module_file('_comparison_patch_helpers', folder / 'PatchTST_layers.py')
        revin = module_file('_comparison_patch_revin', folder / 'RevIN.py')
        old = {name: sys.modules.get(name) for name in ('layers.PatchTST_layers', 'layers.RevIN')}
        sys.modules['layers.PatchTST_layers'] = helpers
        sys.modules['layers.RevIN'] = revin
        try:
            backbone = module_file('_comparison_patch_backbone', folder / 'PatchTST_backbone.py')
        finally:
            for name, module in old.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        patch = cfg['patch_lens'][0]
        self.seq_len = cfg['seq_len']; self.channels = cfg['enc_in']
        self.network = backbone.PatchTST_backbone(
            c_in=self.channels, context_window=self.seq_len, target_window=cfg['pred_len'],
            patch_len=patch, stride=options['stride'], n_layers=cfg['e_layers'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'], d_ff=cfg['d_ff'],
            dropout=cfg['dropout'], attn_dropout=cfg['dropout'], head_dropout=cfg['dropout'],
            padding_patch=None, revin=True, affine=True, res_attention=True,
            individual=False)
        if (self.seq_len-patch) % options['stride']:
            raise ValueError('PatchTST must cover the supplied real window without padding')

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        if x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError('Expected historical input only')
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class PhyGNNForecast(nn.Module):
    """PhyGNN physical-calculation/adapt/aggregate/compensate forecasting adapter.

    Uses the author's G3CN block unchanged, real historical equation nodes and
    the same fixed sensor graph. Temporal heads are added because the released
    SPS model reconstructs a supplied operating segment rather than forecasting
    an unknown future segment. No closed industrial simulator is invented.
    """
    def __init__(self, cfg, data, options):
        super().__init__()
        source = HERE / 'vendor/PhyGNN/PhyGNN/spatial_block.py'
        spatial = module_file('_comparison_phygnn_spatial', source)
        self.channels = data['graph']['channels']; c = len(self.channels)
        self.seq_len = cfg['seq_len']; self.pred_len = cfg['pred_len']
        self.register_buffer('adjacency', augmented_adjacency(data['graph']) + torch.eye(c + len(FEATURES)))
        for key in ('raw_mean', 'raw_scale', 'eq_mean', 'eq_scale'):
            self.register_buffer(key, torch.tensor(data['scalers'][key], dtype=torch.float32))
        self.graph = spatial.CMTS_GCN(options['graph_heads'], c+len(FEATURES),
                                     CMTS_GCN_residual=.1)
        hidden = options['temporal_hidden']
        self.correction = nn.Sequential(nn.Linear(2*self.seq_len, hidden), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(hidden, self.pred_len))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # Target-scale physical soft measurements use the existing train-only
        # observation gains. Do not interpret these as identified device constants.
        route = torch.zeros(c, len(FEATURES)); counts = torch.zeros(c)
        for edge, gain in zip(data['graph']['edges'], data['scalers']['gains']):
            if edge['feature'] is not None:
                route[edge['target'], FEATURES.index(edge['feature'])] += gain/self.raw_scale[edge['target']]
                counts[edge['target']] += 1
        route /= counts.clamp_min(1)[:, None]
        self.register_buffer('physical_route', route)
        self.physical_forecast = nn.Linear(self.seq_len, self.pred_len, bias=False)
        nn.init.zeros_(self.physical_forecast.weight)
        with torch.no_grad():
            self.physical_forecast.weight[:, -1] = 1.

    def forward(self, x, x_mark=None, x_dec=None, x_dec_mark=None):
        if x.shape[1:] != (self.seq_len, len(self.channels)):
            raise ValueError('Expected historical input only')
        raw = x*self.raw_scale + self.raw_mean
        eq = equation_features(raw, self.channels)
        equation_inputs = (eq-self.eq_mean)/self.eq_scale
        augmented = torch.cat((x, equation_inputs), -1).transpose(1, 2)
        graph_history = self.graph(augmented, self.adjacency)[:, :len(self.channels)]
        local_mean = x.mean(1, keepdim=True).detach()
        local_scale = (x.var(1, keepdim=True, unbiased=False)+1e-5).sqrt()
        own = ((x-local_mean)/local_scale).transpose(1, 2)
        # The known equation is evaluated in raw units. Unknown absolute offsets
        # are anchored to the observed last target; future evolution is learned.
        variations = eq-eq[:, -1:]
        mapped = torch.einsum('btf,cf->btc', variations, self.physical_route)
        physical_history = (x[:, -1:] + mapped).transpose(1, 2)
        reference = self.physical_forecast(physical_history).transpose(1, 2)
        residual = self.correction(torch.cat((own, graph_history), -1)).transpose(1, 2)
        return reference + local_scale*residual


class PartialPhysicsLoss(nn.Module):
    """Tolerance-aware local equation-response regularization, NOT a closed PINN.

    Uses nominal finite relation lags and fixed train-calibrated observation
    gains. Missing mass/energy balance terms are never set to zero. A 96-point
    increment tolerance is estimated on Train only; weak equations are soft
    proxies, not exact conservation laws. Future equations use predictions,
    never future measured sources or simulator states.
    """
    def __init__(self, data):
        super().__init__()
        self.channels = data['graph']['channels']; self.length = data['config']['seq_len']
        for key in ('raw_mean', 'raw_scale'):
            self.register_buffer(key, torch.tensor(data['scalers'][key], dtype=torch.float32))
        raw = data['raw_train']
        with torch.no_grad():
            bank = equation_features(torch.as_tensor(raw, dtype=torch.float64), self.channels).numpy()
        routes = {}; rows = []
        for edge, gain in zip(data['graph']['edges'], data['scalers']['gains']):
            if edge['feature'] is None:
                continue
            fi = FEATURES.index(edge['feature']); target = edge['target']
            key = (fi, target)
            if key in routes:
                continue
            routes[key] = True
            lags = sorted(edge['lags']); lag = int(lags[len(lags)//2])
            t = np.arange(self.length+lag, len(raw))
            error = ((raw[t, target]-raw[t-self.length, target]) -
                     gain*(bank[t-lag, fi]-bank[t-self.length-lag, fi]))
            scale = float(data['scalers']['raw_scale'][target])
            tolerance = max(float(np.quantile(np.abs(error), .90))/scale, .02)
            rows.append(dict(feature=edge['feature'], target=self.channels[target],
                             lag=lag, gain=float(gain), tolerance=tolerance,
                             meaning='train-calibrated delayed observation proxy; not an exact balance'))
        self.routes = rows
        self.register_buffer('feature', torch.tensor([FEATURES.index(r['feature']) for r in rows], dtype=torch.long))
        self.register_buffer('target', torch.tensor([self.channels.index(r['target']) for r in rows], dtype=torch.long))
        self.register_buffer('lag', torch.tensor([r['lag'] for r in rows], dtype=torch.long))
        self.register_buffer('gain', torch.tensor([r['gain'] for r in rows], dtype=torch.float32))
        self.register_buffer('tolerance', torch.tensor([r['tolerance'] for r in rows], dtype=torch.float32))

    def forward(self, history, prediction):
        combined = torch.cat((history, prediction), 1)*self.raw_scale + self.raw_mean
        bank = equation_features(combined, self.channels)
        # Each route aligns within the provided historical+predicted segment.
        t = torch.arange(prediction.shape[1], device=prediction.device)
        read = self.length+t[:, None]-self.lag[None]
        initial = self.length-1-self.lag
        source_delta = bank[:, read, self.feature[None]] - bank[:, initial, self.feature][:, None]
        expected_delta = source_delta*self.gain[None, None]/self.raw_scale[self.target][None, None]
        predicted_delta = prediction[:, :, self.target]-history[:, -1:, self.target]
        excess = F.relu((predicted_delta-expected_delta).abs()-self.tolerance[None, None])
        proxy_loss = excess.square().mean() if len(self.routes) else prediction.sum()*0
        # Reported mean cell voltage cannot exceed the reported maximum. This
        # is an observable order constraint, not an invented polarization model.
        ix = {name: i for i, name in enumerate(self.channels)}
        if 'avg' in ix and 'max' in ix:
            future = combined[:, self.length:]
            scale = self.raw_scale[ix['avg']].clamp_min(1e-6)
            proxy_loss = proxy_loss + F.relu((future[..., ix['avg']]-future[..., ix['max']])/scale).square().mean()
        return proxy_loss


def make_model(kind, cfg, data, seed, options):
    seed_all(seed)
    if kind == 'MTGNN':
        return MTGNNForecast(cfg, options, seed)
    if kind == 'PatchTST':
        return PatchTSTForecast(cfg, options)
    if kind == 'PhyGNN':
        return PhyGNNForecast(cfg, data, options)
    if kind == 'PG-Reg-CTF':
        source = ROOT / 'runs/paper_minimal_v1/controls.py'
        controls = module_file('_comparison_equation_input', source)
        base = OriginalCTF(SimpleNamespace(**copy.deepcopy(cfg)))
        fs = data['feature_scaler']
        def bank(raw):
            values = equation_features(raw, data['graph']['channels'])
            return values, torch.isfinite(values)
        return controls.EquationInputCTF(SimpleNamespace(**copy.deepcopy(cfg)), base, bank,
            data['scalers']['raw_mean'], data['scalers']['raw_scale'], fs['mean'], fs['scale'])
    raise ValueError(kind)
