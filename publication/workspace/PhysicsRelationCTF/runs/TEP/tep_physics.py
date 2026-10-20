"""Observable TEP equation adapter; constants from Downs/Vogel teprob.f.

Measurements are noisy, analyzers delayed, and XMV records commands rather than
valve positions. These equations are soft sensor relations, not exact constraints
on every measured row. No simulator hidden state or future observation is used.
"""
import numpy as np
import torch

SOURCE = 'https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f'
RELATIONS = ['valve_flow', 'pressure_flow', 'compressor', 'heat_transfer',
             'cooling_balance', 'liquid_density']
MW = [2., 25.4, 28., 32., 46., 48., 62., 76.]
CONV = .359 / 35.3145


def build_graph(channels):
    ix = {name: i for i, name in enumerate(channels)}
    m = lambda k: f'xmeas_{k}'
    u = lambda k: f'xmv_{k}'
    features, edges = [], []
    def add(name, target, sources, relation, equation, lines, limitations, lags):
        idx = len(features)
        features.append(dict(name=name, op=name, target=ix[target],
            inputs=[ix[s] for s in sources], equation=equation,
            source=SOURCE, source_lines=lines, limitation=limitations,
            output_units='same native units as target'))
        for s in sources:
            edges.append(dict(source=ix[s], target=ix[target], source_name=s,
                target_name=target, relation=RELATIONS.index(relation),
                relation_type=relation, feature=idx, known_gain=1.,
                absolute_reference=True, lags=lags, equation=equation,
                evidence=f'{SOURCE}#L{lines[0]}', limitation=limitations))
    valve_note = 'XMV command approximates unobserved valve position; 5-9 s actuator time constant vs 180 s sampling; noisy measurements.'
    for name, target, source, eq in [
        ('feed_A', m(1), u(3), 'F_A = u3 * (100/100) * 0.359/35.3145'),
        ('feed_D', m(2), u(1), 'F_D = u1 * (400/100) * MW_Dfeed * 0.454'),
        ('feed_E', m(3), u(2), 'F_E = u2 * (400/100) * MW_Efeed * 0.454'),
        ('feed_AC', m(4), u(4), 'F_AC = u4 * (1500/100) * 0.359/35.3145')]:
        add(name,target,[source],'valve_flow',eq,[565,569],valve_note,[0,1,2])
    analyzer = 'Gas composition analyzer has 6-minute sample/delay; use last observed composition, not hidden current composition.'
    gas = [m(k) for k in range(29,37)]
    add('purge',m(10),[u(6),m(13)]+gas,'pressure_flow',
        'Fpurge = 0.151169*u6*sqrt(Psep_mmHg-760)/MWsep * 0.359/35.3145',
        [585,589],valve_note+' '+analyzer,[0,1,2])
    add('recycle',m(5),[u(5),m(16),m(13)]+gas,'compressor',
        'Frec = max(CPFLMX+CPFLMX/1.197*(1-PR^3)-53.349*u5*sqrt(Pv-Ps),0)/MWsep * 0.359/35.3145',
        [590,603],analyzer+' Compressor constants are simulator nominal values.',[0,1,2])
    add('compressor_work',m(20),[m(16),m(13),m(11)]+gas,'compressor',
        'W = FLMS*(Ts+273.15)*1.8e-6*1.9872*(Pv-Ps)/(MWsep*Ps)*293.07',
        [590,596],analyzer,[0,1,2])
    add('reactor_feed',m(6),[m(16),m(7)],'pressure_flow',
        'Ffeed = k_flow * sqrt(max(Pv_kPa-Pr_kPa,0)); k_flow estimated on Train only',
        [576,580],'Full feed mixture molecular weight unobserved: one effective nonnegative coefficient; no invented component measurements.',[0,1,2])
    add('steam',m(19),[u(9),m(18)],'heat_transfer',
        'Fsteam = u9*(0.03/100)*max(100-Tstrip,0)*1040*0.454',
        [572,572],'Steam actuator has 120 s time constant; quasi-static command approximation.',[0,1,2,4])
    add('product_flow',m(17),[u(8),m(18)]+[m(k) for k in range(37,42)],'liquid_density',
        'Fprod = (u8*1000/100)/35.3145 * sum_k(z_k MW_k/(AD_k+BD_k*T+CD_k*T^2))',
        [1490,1504],'Liquid composition D-H renormalized; unobserved trace light species omitted; analyzer delay 15 minutes.',[0,1,2,4])
    add('reactor_cooling',m(21),[u(10),m(9),m(8)],'cooling_balance',
        'Twr_eq = (Fwr*500.53*35 + UAr*1e6/1.8*Tr)/(Fwr*500.53+UAr*1e6/1.8)',
        [663,674],'Quasi-steady cooling-water balance; nominal inlet 35 C and agitator 50%; no inlet fault claimed.',[0,1,2,4,8])
    add('separator_cooling',m(22),[u(11),m(9)],'cooling_balance',
        'Tws_eq = (Fws*500.53*40 + UAeff*1e6/1.8*Tr)/(Fws*500.53+UAeff*1e6/1.8)',
        [674,677],'Unobserved reactor outlet molar flow: effective UA estimated from Train, bounded by simulator maximum 0.404655; nominal inlet 40 C.',[0,1,2,4,8])
    return dict(channels=list(channels), relations=RELATIONS,features=features,
        edges=edges,base_feature_count=len(features),adjacency_convention='A[target,source]',
        lag_units='180-second native samples; candidate alignment ranges, not known process delays',
        excluded=['Hidden-state reaction kinetics: reactor partial pressures unobserved.',
                  'Exact total/component/energy balances require unobserved reactor outlet, vapor return, inventories and enthalpies.',
                  'No pairwise correlations used as physical topology; no transitive graph powers.'])


def equation_bank(raw, graph):
    """Return [...,12] native-unit target estimates and domain validity masks."""
    ix = {c:i for i,c in enumerate(graph['channels'])}
    def m(k): return raw[...,ix[f'xmeas_{k}']]
    def u(k): return raw[...,ix[f'xmv_{k}']].clamp(0,100)
    z = torch.stack([m(k) for k in range(29,37)],-1).clamp_min(0)
    mw = (z*raw.new_tensor(MW)).sum(-1)/z.sum(-1).clamp_min(1e-6)
    mw = mw.clamp(2,76)
    ps=(m(13)/101.325+1)*760
    pv=(m(16)/101.325+1)*760
    dp=(pv-ps).clamp_min(0)
    pr=(pv/ps.clamp_min(1)).clamp(1,1.3)
    mass=280275.+(280275./1.197)*(1-pr.pow(3))
    zl=torch.stack([m(k) for k in range(37,42)],-1).clamp_min(0)
    zl=zl/zl.sum(-1,keepdim=True).clamp_min(1e-6)
    t=m(18).unsqueeze(-1)
    density_denom=(raw.new_tensor([23.3,33.9,32.8,49.9,50.5])+
        raw.new_tensor([-.0700,-.0957,-.0995,-.0191,-.0541])*t+
        raw.new_tensor([-.0002,-.000152,-.000233,-.000425,-.000150])*t*t)
    molar_volume=(zl*raw.new_tensor(MW[3:])/density_denom.clamp_min(1.)).sum(-1)
    vlr=84.6+666.7*m(8)/100
    ua=.855490*(.025*vlr/7.8-.25).clamp(0,1)
    ar=u(10)*10*500.53; br=ua*1e6/1.8
    asep=u(11)*12*500.53; bsep=graph.get('ua_separator',.3)*1e6/1.8
    values=dict(
        feed_A=u(3)*CONV,
        feed_D=u(1)*4*(.9999*32+.0001*25.4)*.454,
        feed_E=u(2)*4*(.9999*46+.0001*48)*.454,
        feed_AC=u(4)*15*CONV,
        purge=u(6)*.151169*(ps-760).clamp_min(0).sqrt()/mw*CONV,
        recycle=(mass-u(5)*53.349*dp.sqrt()).clamp_min(.001)/mw*CONV,
        compressor_work=mass*(m(11)+273.15)*1.8e-6*1.9872*dp/(mw*ps.clamp_min(1))*293.07,
        reactor_feed=graph.get('k_flow',1.)*(m(16)-m(7)).clamp_min(0).sqrt(),
        steam=u(9)*.0003*(100-m(18)).clamp_min(0)*1040*.454,
        product_flow=u(8)*10*molar_volume/35.3145,
        reactor_cooling=(ar*35+br*m(9))/(ar+br).clamp_min(1),
        separator_cooling=(asep*40+bsep*m(9))/(asep+bsep).clamp_min(1))
    q=torch.stack([values[f['name']] for f in graph['features']],-1)
    valid=torch.isfinite(q)
    for k,f in enumerate(graph['features']):
        if f['name'] in ('purge','recycle','compressor_work'):
            valid[...,k] &= (ps>760)&(z.sum(-1)>80)&(z.sum(-1)<120)
        if f['name'] in ('recycle','compressor_work'):
            valid[...,k] &= (pv>=ps)
        if f['name']=='reactor_feed':valid[...,k] &= m(16)>m(7)
        if f['name']=='product_flow':valid[...,k] &= (density_denom>1).all(-1)
    return torch.nan_to_num(q),valid


def calibrate(raw_train, graph, mean, scale):
    """Only two unobserved effective coefficients; all other constants fixed."""
    ix={c:i for i,c in enumerate(graph['channels'])}
    m=lambda k: raw_train[:,ix[f'xmeas_{k}']]
    dp=np.maximum(m(16)-m(7),1e-8)
    # Effective inverse molecular weight: robust positive calibration.
    graph['k_flow']=float(np.median(m(6)/np.sqrt(dp)))
    a=raw_train[:,ix['xmv_11']]*12*500.53
    ua=a*(m(22)-40)/np.maximum(m(9)-m(22),1e-3)*1.8e-6
    graph['ua_separator']=float(np.clip(np.median(ua),1e-4,.404655))
    with torch.no_grad():
        q,v=equation_bank(torch.tensor(raw_train,dtype=torch.float64),graph)
    q,v=q.numpy(),v.numpy()
    metrics=[]
    for k,f in enumerate(graph['features']):
        mask=v[:,k];target=f['target'];err=(q[mask,k]-raw_train[mask,target])/scale[target]
        metrics.append(dict(name=f['name'],target=graph['channels'][target],
            normalized_mse=float(np.mean(err**2)),normalized_bias=float(np.mean(err)),
            valid_fraction=float(mask.mean()),correlation=float(np.corrcoef(q[mask,k],raw_train[mask,target])[0,1])))
    return dict(mean=mean.tolist(),scale=scale.tolist(),gains=[1.]*len(graph['edges']),kappa=1.),metrics
