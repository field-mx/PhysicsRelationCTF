"""Report fixed-test results; never chooses models or favorable channels/windows."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parent


def main():
    summary=json.loads((ROOT/'final/summary.json').read_text())
    selected=summary['selection'];names=[selected['ctf'],selected['physics']]
    labels=['Original CTF','TEP PhysicsRelationCTF']
    arrays=[np.load(ROOT/'final'/n/'predictions.npz') for n in names]
    if not np.array_equal(arrays[0]['origins'],arrays[1]['origins']):raise AssertionError('Mismatched test origins')
    np.testing.assert_allclose(arrays[0]['truth'],arrays[1]['truth'],rtol=0,atol=0)
    channels=arrays[0]['channels'];origins=arrays[0]['origins'];scale=arrays[0]['scale'];mean=arrays[0]['mean']
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    # Fixed first official test run; no selection based on performance.
    run=origins[:,0].min();mask=origins[:,0]==run
    truth=arrays[0]['truth'][mask].reshape(-1,len(channels))*scale+mean
    preds=[a['prediction'][mask].reshape(-1,len(channels))*scale+mean for a in arrays]
    times=np.concatenate([np.arange(t,t+96) for _,t in origins[mask]])*3/60
    with PdfPages(figdir/'all_channels_full_test.pdf') as pdf:
        for c,ch in enumerate(channels):
            fig,axes=plt.subplots(2,1,figsize=(11,6),gridspec_kw={'height_ratios':[2,1]})
            for ax,k in zip(axes,[len(times),96]):
                ax.plot(times[:k],truth[:k,c],color='black',lw=.8,label='Observed')
                for p,label,color in zip(preds,labels,['#3975b6','#d87528']):ax.plot(times[:k],p[:k,c],lw=1,label=label,color=color)
                ax.grid(alpha=.2);ax.set_ylabel('Native measurement units');ax.set_xlabel('Time in simulation (hours)')
            axes[0].set_title(f'{ch}: first fixed test run; 96 input / 96 forecast, stride 96')
            axes[0].legend(ncol=3,fontsize=8);axes[1].set_title('First complete 96-step forecast (4.8 hours)')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    metrics=[summary['results'][n] for n in names]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,key in zip(axes,['mse','mae']):
        vals=[m[key] for m in metrics];bars=ax.bar(labels,vals,color=['#3975b6','#d87528'])
        ax.bar_label(bars,fmt='%.6f');ax.set_ylabel('Train-standardized '+key.upper());ax.set_ylim(0,max(vals)*1.15)
    fig.tight_layout();fig.savefig(figdir/'accuracy_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for metric,label in zip(metrics,labels):ax.plot(np.arange(1,97)*3,metric['horizon_mse'],label=label)
    ax.set(xlabel='Forecast lead time (minutes)',ylabel='Standardized MSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'horizon_comparison.png',dpi=180);plt.close(fig)
    gain=np.array(metrics[0]['per_channel_mse'])-np.array(metrics[1]['per_channel_mse'])
    fig,ax=plt.subplots(figsize=(9,12));ax.barh(channels,gain,color=np.where(gain>0,'#3b8a62','#b75959'))
    ax.axvline(0,color='black',lw=.7);ax.set_xlabel('MSE(CTF) - MSE(Physics); positive favors physics')
    fig.tight_layout();fig.savefig(figdir/'per_channel_accuracy.png',dpi=180);plt.close(fig)
    # Paired uncertainty across independently simulated test runs, not windows.
    squared=[(a['prediction'].astype('float64')-a['truth'])**2 for a in arrays]
    deltas=np.array([(squared[0][origins[:,0]==r]-squared[1][origins[:,0]==r]).mean() for r in np.unique(origins[:,0])])
    rng=np.random.default_rng(42);boot=deltas[rng.integers(0,len(deltas),(10000,len(deltas)))].mean(1)
    ci=np.quantile(boot,[.025,.975])
    (ROOT/'final/paired_run_uncertainty.json').write_text(json.dumps(dict(mean_mse_difference=float(deltas.mean()),
        interval95=ci.tolist(),test_runs=len(deltas),physics_better_runs=int((deltas>0).sum()),
        limitation='Fixed trained checkpoints; does not include training-seed variation.'),indent=2))
    rows=['|模型|Test MSE|Test MAE|','|---|---:|---:|']
    rows += [f'|{label}|{m["mse"]:.6f}|{m["mae"]:.6f}|' for label,m in zip(labels,metrics)]
    candidates=['|候选|Val MSE|Val MAE|最佳轮次|','|---|---:|---:|---:|']
    candidates += [f'|{r["name"]}|{r["mse"]:.6f}|{r["mae"]:.6f}|{r["best_epoch"]}|' for r in selected['candidates']]
    text=f'''# TEP 方程适配实验结果

本实验保持 V7 的条件物理消息、可信度抑制和有界残差范式，重新建立 TEP 方程库。
所有模型从 Channel Token 解码。不是使用电解槽或楼宇方程预测 TEP。

## 固定任务

- 52 个测量/操纵通道；80 条训练、20 条验证、100 条测试轨迹。
- 100% 指固定缓存训练池全部 80 条轨迹，不是原始公开数据全部运行。
- 输入 96 点、输出 96 点；每点 180 秒，预测时长 4.8 小时。
- 训练步长 1、验证/测试步长 96，不跨轨迹建窗。
- 固定种子 42、共享 CTF 初始化、Adam、batch 64、最多16轮，相同早停规则。
- 训练统计归一化；评价覆盖全部通道、完整预测时距。测试配置冻结后才读取测试目标。

## 验证选择

{chr(10).join(candidates)}

CTF 选择 `{selected['ctf']}`；Physics 选择 `{selected['physics']}`。
同时提供相同学习率对照 `{selected['matched_ctf']}`。

## 测试结果

{chr(10).join(rows)}

MSE 改善 {summary['mse_gain_percent']:.4f}%；MAE 改善 {summary['mae_gain_percent']:.4f}%。
改善通道 {summary['improved_channels']}/52；改善预测步 {summary['improved_horizons']}/96。
按独立测试轨迹配对重采样，MSE差值95%区间 [{ci[0]:.6f}, {ci[1]:.6f}]。
该区间不包含训练随机种子的变异，不能据此宣称多种子稳定性。

## 物理适配

`tep_physics.py` 提供12类观测方程，`physics_graph.json` 逐条记录直接边、方程、来源和近似条件。
进料阀特性、压差流动、压缩机功率/回流、蒸汽换热、液体密度与冷却水平衡来自公开模拟器。
仅有效压差流量系数和分离器换热系数在训练集标定，其余模拟器常数固定。
所有历史方程在目标量纲下编码；历史不一致的方程通过 V7 可信度衰减。
若选择 closure 候选，输出物理头只读取网络预测的未来变量及历史可信度，绝不读取真实未来控制量。

## 适用边界

TEP 是仿真数据，本次只有正常无故障工况。控制指令不是阀门实际位置，组分分析仪存在采样延迟；
部分物理关系是准稳态近似，可靠性已在训练数据诊断中完整记录。
成功不代表故障工况、未知工况、小样本或真实装置均受益；单种子结果需要后续复现。
旧 TEP 测试集此前已用于其他结构评估，本轮全部选择依据独立验证数据，但它并非从未使用过的新测试集。

图像位于 `figures/`。波形图固定展示官方测试池第一条轨迹，未挑选有利窗口。
原始预测与指标位于 `final/`，每次训练的配置、轨迹和检查点位于 `development/`。

来源：https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f
数据：https://doi.org/10.7910/DVN/6C3JR1
'''
    (ROOT/'RESULTS.md').write_text(text)
    print(text)


if __name__=='__main__':main()
