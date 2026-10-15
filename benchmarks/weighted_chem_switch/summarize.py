#!/usr/bin/env python3
"""汇总独立 GPU 与完整 LAMMPS 测量，保留原始重复样本和数值验证。"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import platform
from datetime import date
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", type=Path, required=True)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((args.micro / "micro.csv").open()))
    keys = sorted({(int(row["atoms"]), int(row["centers"]), int(row["neighbors"]), row["phase"]) for row in rows})
    summaries = []
    for atoms, centers, neighbors, phase in keys:
        selected = [row for row in rows if (int(row["atoms"]), int(row["centers"]), int(row["neighbors"]), row["phase"]) == (atoms, centers, neighbors, phase)]
        samples = {variant: [float(row[variant+"_us"]) for row in selected] for variant in ("separate", "fused")}
        medians = {variant: statistics.median(values) for variant, values in samples.items()}
        summaries.append(dict(atoms=atoms, centers=centers, neighbors=neighbors, phase=phase,
                              median_us=medians, fused_speedup=medians["separate"]/medians["fused"],
                              paired_speedup_median=statistics.median(float(row["speedup"]) for row in selected),
                              sample_ranges={variant: [min(values), max(values)] for variant, values in samples.items()}))
    micro_log = (args.micro/"micro.log").read_text()
    assert "SWITCH_MICRO_PASS" in micro_log
    validation = [dict(atoms=int(atoms), centers=int(centers), neighbors=int(neighbors),
                       value_error=float(value), force_error=float(force), virial_error=float(virial))
                  for atoms, centers, neighbors, value, force, virial in re.findall(
                      r"VALID atoms=(\d+) centers=(\d+) neighbors=(\d+) value_error=(\S+) force_error=(\S+) virial_error=(\S+)", micro_log)]
    assert len(validation) == len({key[:3] for key in keys}) and rows
    lammps = json.loads((args.lammps/"summary.json").read_text())
    (args.output/"micro_summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    (args.output/"micro_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    shutil.copy2(args.micro/"micro.csv", args.output/"micro.csv")
    for name in ("summary", "results", "validation"):
        shutil.copy2(args.lammps/(name+".json"), args.output/("lammps_"+name+".json"))
    metadata = json.loads((args.micro/"metadata.json").read_text())
    root = Path(__file__).resolve().parents[2]
    assert all(hashlib.sha256((root/filename).read_bytes()).hexdigest() == digest
               for filename, digest in metadata["production_sha256"].items())
    metadata["plugins"] = {variant: hashlib.sha256(path.read_bytes()).hexdigest() for variant, path in {
        "separate": root/"build/graph-lammps/fix_crystallize_plugin.so",
        "fused": args.micro/"fused/fix_crystallize_plugin.so"}.items()}
    metadata["measured_environment"] = dict(
        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
        cuda=subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True).strip(),
        cpu=[line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")][0].split(":",1)[1].strip(),
        platform=platform.platform(), mpi_ranks_share_gpu=True)
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    timed_rows = json.loads((args.lammps/"results.json").read_text())
    warmup_input = (Path(timed_rows[0]["directory"])/"input.lmp").read_text()
    warmup_steps = int(re.findall(r"^run (\d+)$", warmup_input, re.M)[-2])
    repetitions = len({row["repeat"] for row in timed_rows})
    report = [f"# SW_func 分离 / 内联测量（{date.today()}）", "",
              "单位均为微秒，时间取重复样本的中位数。加速比为分离时间 / 内联时间，大于 1 表示内联更快。", "",
              "## GPU 前向 + 反向", "",
              "| 总原子数 | 中心数 | 邻居数/中心 | 分离 | 内联 | 加速比 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["phase"] == "pair":
            report.append(f'| {row["atoms"]} | {row["centers"]} | {row["neighbors"]} | {row["median_us"]["separate"]:.2f} | {row["median_us"]["fused"]:.2f} | {row["fused_speedup"]:.3f} |')
    report += ["", f"各阶段、配对比值和最小/最大样本见 micro_summary.json，全部 {len(rows)} 个样本见 micro.csv。许多场景的样本区间重叠，个位数百分比的变化应谨慎解释。", "",
               "## 完整 LAMMPS timestep", "",
               f"{timed_rows[0]['atoms']:,} 原子，{warmup_steps:,} 步预热，再运行 {timed_rows[0]['steps']:,} 步，每种配置重复 {repetitions} 次，随机化两版运行顺序。", "",
               "| MPI ranks | 分离 µs/step | 内联 µs/step | 加速比 | 分离样本范围 | 内联样本范围 |",
               "|---:|---:|---:|---:|---:|---:|"]
    for row in lammps:
        ranges = {variant: "–".join(f"{value:.1f}" for value in values) for variant, values in row["sample_ranges"].items()}
        report.append(f'| {row["ranks"]} | {row["median_us"]["separate"]:.2f} | {row["median_us"]["fused"]:.2f} | {row["fused_speedup"]:.3f} | {ranges["separate"]} | {ranges["fused"]} |')
    report += ["", "NVRTC、初始化与预热均排除在计时之外。完整流程仍包含 Graph 装配、AVE、Host/GPU 数据传输、MPI、力及 virial；内联版将 Fermi 放入 CV kernel，图处理 AVE。", "",
               "## 正确性", "",
               f'{len(validation)} 组 GPU 场景全部通过；值最大差 {max(row["value_error"] for row in validation):.3g}，力最大差 {max(row["force_error"] for row in validation):.3g}，virial 最大差 {max(row["virial_error"] for row in validation):.3g}。',
               "完整 LAMMPS 另在 1/2 rank 验证非零受力、CV、能量、位置和六分量压力，误差见 lammps_validation.json。", "",
               "生产源码哈希在实验前后保持一致；内联对照的源文件及插件均在 build 中。环境、源码及插件哈希见 metadata.json。", ""]
    (args.output/"RESULTS.md").write_text("\n".join(report))
    print(json.dumps(lammps, indent=2))


if __name__ == "__main__":
    main()
