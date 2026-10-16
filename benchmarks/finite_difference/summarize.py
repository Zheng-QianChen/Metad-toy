#!/usr/bin/env python3
"""归档差分曲线、MPI 一致性与线性叠加结果；原始逐坐标明细留在 build。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();source=args.input.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    results=json.loads((source/'full/results.json').read_text())
    linear=json.loads((source/'full/linear_results.json').read_text())
    graph=[]
    for line in (source/'ctest.log').read_text().splitlines():
        if 'GRAPH_FD_RESULT ' in line:graph.append(json.loads(line.split('GRAPH_FD_RESULT ',1)[1]))
    assert len(graph)==18 and '100% tests passed, 0 tests failed out of 19' in (source/'ctest.log').read_text()
    mpi=[]
    for row in results:
        baseline=source/'full'/f"{row['case']}_np1/state.json"
        current=source/'full'/f"{row['case']}_np{row['ranks']}/state.json"
        if row['ranks']==1 or not baseline.exists() or not current.exists():continue
        before=json.loads(baseline.read_text());after=json.loads(current.read_text())
        errors={key:float(np.max(np.abs(np.array(after[key])-np.array(before[key])))) for key in ['rows','values','virial']}
        errors['rank_value_spread']=float(np.ptp(np.asarray(row['rank_values']),axis=0).max())
        mpi.append(dict(case=row['case'],ranks=row['ranks'],passed=max(errors.values())<1e-10,**errors))
    write=lambda name,value:(output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    write('results.json',results);write('linear_results.json',linear);write('graph_results.json',graph);write('mpi_results.json',mpi)
    diagnostics=source/'distance-diagnostic/distance_linear_np2/diagnostics.json'
    write('distance_diagnostics.json',json.loads(diagnostics.read_text()))
    good=[row for row in results if row['passed']]
    maxima={key:max(row['curves'][-1][key]['max_abs_error'] for row in good if key in row['curves'][-1])
            for key in ['force','virial','cv_gradient','cv_strain']}
    maxima['linear_cv_gradient']=max(row['cv_gradient_error'] for row in linear)
    maxima['graph_nvrtc']=max(row['max_abs_error'] for row in graph if row['case']=='nvrtc_mpi' and row['step']<2e-5)
    maxima['graph_host']=max(row['max_abs_error'] for row in graph if row['case']=='host' and row['step']<2e-5)
    files=list((ROOT/'src/cvs').glob('*.cu'))+list((ROOT/'src/utils').glob('zqc_graph*.cu'))+[
        ROOT/'include/utils/zqc_graph_device.cuh',ROOT/'src/core/fix_crystallize.cu',
        ROOT/'benchmarks/finite_difference/graph.cu',ROOT/'benchmarks/finite_difference/bias_probe.cpp']
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=dict(date='2026-10-16',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plugin_sha256=json.loads((source/'full/metadata.json').read_text())['plugin_sha256'],
        validation_directory=str(source),cv_results=dict(total=len(results),passed=len(good),failed=len(results)-len(good)),
        linear_checks=len(linear),ctest='19/19 passed',max_errors_at_smallest_step=maxima,
        full_sweep_runner_sha256=digest(source/'run_full_snapshot.py'),
        current_runner_sha256=digest(ROOT/'benchmarks/finite_difference/run.py'),
        runner_change='After full sweep, added nonfinite-force diagnostics; reran failing DISTANCE linear np2 only.',
        source_sha256={str(path.relative_to(ROOT)):digest(path) for path in files})
    write('metadata.json',metadata)
    lines=['# 有限差分结果（2026-10-16）','',
           '完整 LAMMPS 检查 60 组：Graph 路径 54 组通过，旧 DISTANCE 相关 6 组失败。线性叠加对照 9/9 通过；独立 Graph 与既有 CTest 合计 19/19 通过。','',
           '下表误差取最小步长 `h=1e-5`、1/2/4 rank 中的最大值。CV 导数从单输出偏置力除以非零 `dV/ds` 得到；非线性双输出只比较总偏置能的导数。所有数值使用夹具的 LAMMPS LJ 单位。','',
           '| 场景 | 通过数 | CV 坐标导数误差 | CV 应变导数误差 | 偏置力误差 | 偏置 virial 误差 |',
           '|---|---:|---:|---:|---:|---:|']
    for case in dict.fromkeys(row['case'] for row in results):
        selected=[row for row in results if row['case']==case]
        values=[]
        for key in ['cv_gradient','cv_strain','force','virial']:
            errors=[row['curves'][-1][key]['max_abs_error'] for row in selected if row.get('curves') and key in row['curves'][-1]]
            values.append(f'{max(errors):.3e}' if errors else '—')
        lines.append(f"| {case} | {sum(row['passed'] for row in selected)}/{len(selected)} | "+' | '.join(values)+' |')
    lines+=['','## Graph 与线性组合','',
        f"- 独立 NVRTC/MPI Graph：19 个逐原子样本、两个输入 Leaf、一个 Global 输入，共 39 个变量、18 个输出。最小步长最大导数误差 `{maxima['graph_nvrtc']:.3e}`。",
        f"- Host Graph：2 个 Global 输入、14 个输出。最小步长最大导数误差 `{maxima['graph_host']:.3e}`。",
        '- 对每个输出分别反向求 Jacobian；另用正负混合输出种子检查 VJP，并验证重复 backward 清零。覆盖共享乘积、共享 AVE、MPI 不均匀分区和空 rank。',
        f"- Q/Local、化学 CV 和全部 Graph CV 的线性组合，共 9 组加权和对照；坐标梯度最大差 `{maxima['linear_cv_gradient']:.3e}`。",
        '- Graph 使用 `h=1e-3,1e-4,1e-5`，前者展示截断误差，后两者均满足阈值。LAMMPS 使用 `h=1e-4,3e-5,1e-5`；通过的 Graph 场景在后两个步长均满足阈值。',
        '', '## DISTANCE 失败定位', '',
        '- 单 rank：距离和坐标导数通过，但 virial 为零，应变差分非零。`Distance::bias_force` 没有向 Fix 的 `a_virial` 累加。',
        '- 多 rank：输入全局 atom ID 被减一后直接用于 `x[]` / `f[]` 本地下标，既未做 owner 查找，也未分发坐标和力。`DISTANCE 1 2` 在 4 rank 上分别得到 2.866122、2.820468、2.912008、2.840903。',
        '- 距离线性组合在 2 rank 上产生非有限力；补充诊断记录了 atom ID 5 的三个力分量为非有限值。这一组在扫描前终止，表格使用其余可完成扫描的失败记录。',
        '- 测试程序保留失败退出码，未将这些问题设为预期成功；修复 DISTANCE 的编号、MPI 和 virial 后可用同一套输入回归。',
        '', '## 原始记录', '',
        '`results.json` 保存所有步长的最大绝对误差、RMS、按绝对/相对阈值归一化的误差。`graph_results.json` 保存每个 Graph 输出的误差，最后一项为混合种子 VJP。`linear_results.json`、`mpi_results.json` 分别记录加权叠加与 rank 一致性。源码与插件摘要见 `metadata.json`。',
        '',f'输入、逐坐标明细、LAMMPS 日志和编译日志位于 `{source}`。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metadata['max_errors_at_smallest_step'],indent=2))
    print('Graph MPI state checks',sum(row['passed'] for row in mpi if not row['case'].startswith('distance')),
          '/',sum(not row['case'].startswith('distance') for row in mpi))


if __name__=='__main__':main()
