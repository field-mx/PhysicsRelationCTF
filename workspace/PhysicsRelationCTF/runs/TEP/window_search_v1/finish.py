"""Wait for the four approved jobs, freeze configs, evaluate and summarize."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import runner

HERE=Path(__file__).resolve().parent


def summarize(summary):
    uncertainty={};rows=[];common=[];val=[]
    all_arrays={}
    for group,setting in runner.GROUPS.items():
        arrays={kind:np.load(HERE/'final'/group/kind/'predictions.npz') for kind in ['CTF','Physics']}
        all_arrays[group]=arrays
        np.testing.assert_array_equal(arrays['CTF']['truth'],arrays['Physics']['truth'])
        np.testing.assert_array_equal(arrays['CTF']['origins'],arrays['Physics']['origins'])
        error={k:(a['prediction'].astype('float64')-a['truth']) for k,a in arrays.items()}
        for k,e in error.items():
            metric=summary['results'][group][k]
            np.testing.assert_allclose([np.mean(e**2),np.mean(abs(e))],[metric['mse'],metric['mae']],rtol=1e-10)
        origin=arrays['CTF']['origins'];runids=np.unique(origin[:,0]);delta=[]
        for r in runids:
            mask=origin[:,0]==r
            delta.append(float((error['CTF'][mask]**2-error['Physics'][mask]**2).mean()))
        delta=np.array(delta);rng=np.random.default_rng(42)
        boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        uncertainty[group]=dict(mse_difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
            improved_runs=int((delta>0).sum()),run_count=len(runids),note='Paired trajectory bootstrap, not training-seed uncertainty.')
        c=summary['comparisons'][group]
        rows.append(f'|{setting["length"]}→{setting["horizon"]}|{c["ctf_mse"]:.6f}|{c["physics_mse"]:.6f}|{c["mse_gain_percent"]:.3f}%|{c["ctf_mae"]:.6f}|{c["physics_mae"]:.6f}|{c["mae_gain_percent"]:.3f}%|')
        common.append(f'|{group}|{c["common10_ctf_mse"]:.6f}|{c["common10_physics_mse"]:.6f}|{c["common10_gain_percent"]:.3f}%|')
        for k in ['CTF','Physics']:
            s=summary['selection'][group][k]
            val.append(f'|{group}|{k}|{s["validation"]["mse"]:.6f}|{s["validation"]["mae"]:.6f}|{s["best_epoch"]}|{s["parameters"]}|')
    a=all_arrays['ratio_20x20']['CTF'];b=all_arrays['paper_50x10']['CTF']
    np.testing.assert_array_equal(a['origins'],b['origins']);np.testing.assert_array_equal(a['truth'][:,:10],b['truth'])
    runner.base.dump(HERE/'final/uncertainty.json',uncertainty)
    lock=json.loads((HERE/'final/frozen_manifest.json').read_text())
    verified={}
    for category in ['code','checkpoints','data']:
        for path,expected in lock[category].items():
            if runner.base.sha(path)!=expected:raise RuntimeError(f'Frozen artifact changed: {path}')
        verified[category]=len(lock[category])
    old=json.loads((HERE.parent/'final/frozen_test_manifest.json').read_text())
    for category in ['code','checkpoints','data']:
        for path,expected in old[category].items():
            if runner.base.sha(path)!=expected:raise RuntimeError(f'Old experiment changed: {path}')
    verified.update(metrics_recomputed=4,common10_targets_identical=True,old_experiment_unchanged=True)
    runner.base.dump(HERE/'final/verification.json',verified)
    diagnostics={}
    for group in runner.GROUPS:
        training=json.loads((HERE/group/'development/Physics/training.json').read_text())
        history=json.loads((HERE/group/'development/Physics/history.json').read_text())
        diagnostics[group]=next(r['diagnostics'] for r in history if r['epoch']==training['best_epoch'])
    runner.base.dump(HERE/'final/selected_physics_diagnostics.json',diagnostics)
    text='''# TEP 真实观测窗口配对实验

## 冻结物理模型与控制变量

采用第一轮 physics_nominal 的12个方程特征、52条物理边与训练标定系数；不采用 optimization_v2 的新反解关系或平滑。两个模型均从 Channel Token 解码。
不复制、不补齐、不插值，不改变原数据文件、52通道及训练/验证/测试轨迹划分。
固定seed42、Adam、lr3e-4、batch64、最多16轮、相同早停和调度；每组公共CTF参数初始值一致，重新训练，不继承不匹配长度的预测头。

|组|输入|输出|patch|每通道local tokens|预测时间|
|---|---:|---:|---:|---:|---|
|比例参考|20|20|5|4|60分钟|
|文献参考|50|10|10|5|30分钟|

工业数据96/4847约1.981%，本次20/960约2.083%，是整除patch的近似匹配，不是严格同比例。
文献参考来自 Djogap Feujo & Chioua, IFAC-PapersOnLine59(6),2025, DOI10.1016/j.ifacol.2025.07.180，50点历史、10点未来。这里只参考窗口；原论文的故障分类、12变量、截取100点等设定不迁移到当前数据。
论文原文：https://skoge.folk.ntnu.no/prost/proceedings/dycops-2025/DYCOPS2025_offline_program/contents/papers/0085.pdf

## 窗口口径

沿用80训练、20验证、100测试轨迹。两组均采用相同预测起点：训练t=50…480（stride1），验证t=50…470（stride20），测试t=50…930（stride20）。
窗口数量34480/440/4500；每条轨迹独立，不跨边界。
测试的前50点不作为预测目标；尾部20→20剩10点、50→10剩20点不计分。
50→10采用共同stride20，因此预测块之间有10点未计分；这不是删除源数据。共同前10步在完全相同的真实目标上比较。

## 验证结果（测试前锁定）

|组|模型|Val MSE|Val MAE|最佳轮次|参数量|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(val)+f'''

验证集按同任务相对MSE改善选出的设置：`{summary['selection']['preferred_on_validation_relative_gain']}`。所有组均完整报告，不按测试结果反选。

## 原固定测试集：同任务配对比较

|输入→输出|CTF MSE|Physics MSE|MSE改善|CTF MAE|Physics MAE|MAE改善|
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 相同预测起点、共同前10步

|组|CTF MSE|Physics MSE|改善|
|---|---:|---:|---:|
'''+ '\n'.join(common)+f'''

## 不确定性与边界

配对轨迹重采样：`{json.dumps(uncertainty,ensure_ascii=False)}`。
同一组的Physics增益可以比较；两个组的完整预测MSE不能直接用于证明窗口优劣，因为预测时距不同。
前10步对比控制了目标和起点，但同时改变历史长度及patch长度，不是单变量的历史长度消融。
当前仅单种子、正常工况仿真数据。此前已查看过该测试集，属于开发基准，不是新盲测，也不能证明小样本或故障泛化。
只有两个预设设置，不宣称找到全局最优或保证超越CTF。

## 文件

模型：length_model.py；训练/评价：runner.py；协议测试：test_lengths.py、test_windows.py。
权重：各组development/CTF/best.pth及development/Physics/best.pth。
逐通道/逐时距指标和完整预测：final/；冻结记录：final/frozen_manifest.json。
指标复算、跨组目标一致及旧文件未变检查：final/verification.json。
最新用户批准规则见EXECUTION.md，覆盖PLAN.md里未执行的19点/补齐草案。
'''
    (HERE/'RESULTS.md').write_text(text)
    print(json.dumps(summary['comparisons']),flush=True)
    runner.base.dump(HERE/'status.json',dict(state='complete',summary=str(HERE/'RESULTS.md')))


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    required=[HERE/g/'development'/k/'training.json' for g in runner.GROUPS for k in ['CTF','Physics']]
    while not all(p.exists() for p in required):
        for service in ['tep-window-ratio','tep-window-paper']:
            state=subprocess.run(['systemctl','--user','show',service,'-p','Result','--value'],capture_output=True,text=True).stdout.strip()
            if state and state!='success':raise RuntimeError(f'Training service failed: {service}: {state}')
        time.sleep(30)
    summarize(runner.evaluate_final('cuda:1'))


if __name__=='__main__':main()
