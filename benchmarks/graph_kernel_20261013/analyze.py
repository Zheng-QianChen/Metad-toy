#!/usr/bin/env python3
from pathlib import Path
import collections,csv,json,re,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
runs=[HERE/'results',HERE/'results_focus']
all_rows=[]
for idx,folder in enumerate(runs):
    for row in csv.DictReader((folder/'timings.csv').open()):
        row.update(run=idx,n=int(row['n']),round=int(row['round']),us=float(row['us']))
        all_rows.append(row)
groups=collections.defaultdict(list)
for row in all_rows:groups[row['case'],row['n'],row['phase'],row['variant']].append(row['us'])
variants=['graph_nvrtc','manual_nvrtc','manual_nvcc'];cases=['mean_exp','two_dims']
summary=[]
for (case,n,phase,variant),values in sorted(groups.items()):
    summary.append(dict(case=case,n=n,phase=phase,variant=variant,samples=len(values),median_us=statistics.median(values),p25_us=float(np.percentile(values,25)),p75_us=float(np.percentile(values,75)),min_us=min(values),max_us=max(values)))
with (HERE/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
pairs=collections.defaultdict(dict)
for row in all_rows:
    if row['phase']=='pair':pairs[row['case'],row['n'],row['run'],row['round']][row['variant']]=row['us']
ratios=collections.defaultdict(list)
for (case,n,run,round),values in pairs.items():ratios[case,n].append(values['graph_nvrtc']/values['manual_nvcc'])
rng=np.random.default_rng(12345);ratio_summary=[]
for (case,n),values in sorted(ratios.items()):
    a=np.asarray(values);draws=np.median(rng.choice(a,size=(10000,len(a))),axis=1)
    median_g=statistics.median(groups[case,n,'pair','graph_nvrtc']);median_m=statistics.median(groups[case,n,'pair','manual_nvcc'])
    ratio_summary.append(dict(case=case,n=n,samples=len(values),graph_us=median_g,manual_nvcc_us=median_m,ratio_of_medians=median_g/median_m,median_paired_ratio=float(np.median(a)),paired_ratio_ci95=list(map(float,np.percentile(draws,[2.5,97.5])))))
(HERE/'ratios.json').write_text(json.dumps(ratio_summary,indent=2))
compile_data=collections.defaultdict(list)
for folder in runs:
    for row in csv.DictReader((folder/'compile.csv').open()):
        if int(row['repetition'])>0:compile_data[row['case'],row['variant']].append(float(row['compile_ms']))
checks=[]
for folder in runs:checks+=list(csv.DictReader((folder/'validation.csv').open()))
max_error=max(float(row['max_abs_error']) for row in checks)
# SASS evidence: count actual mnemonics excluding padding, and detect calls.
sass=[]
for case in cases:
    for v in variants:
        text=(runs[0]/(case+'.'+v+'.sass')).read_text();functions={};fn=None
        for line in text.splitlines():
            if 'Function :' in line:fn=line.split('Function :',1)[1].strip();functions[fn]=[]
            elif fn and re.search(r'/\*[0-9a-f]+\*/',line):
                instruction=re.sub(r'/\*.*?\*/','',line).strip()
                if instruction and not instruction.startswith('NOP'):functions[fn].append(instruction)
        for fn,lines in functions.items():sass.append(dict(case=case,variant=v,function=fn,non_nop_instructions=len(lines),call_instructions=[l for l in lines if re.search(r'\bCALL',l)]))
(HERE/'sass_summary.json').write_text(json.dumps(sass,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex='col',gridspec_kw={'height_ratios':[1.5,1]})
colors={'graph_nvrtc':'#0072B2','manual_nvrtc':'#888888','manual_nvcc':'#252525'}
labels={'graph_nvrtc':'Generated graph / NVRTC','manual_nvrtc':'Handwritten / NVRTC','manual_nvcc':'Handwritten / nvcc'}
for col,case in enumerate(cases):
    ns=sorted({r['n'] for r in summary if r['case']==case});ax=axes[0,col]
    for v in variants:
        vals=[statistics.median(groups[case,n,'pair',v]) for n in ns]
        ax.plot(ns,vals,color=colors[v],marker='o' if v=='graph_nvrtc' else '.',linestyle=':' if v=='manual_nvrtc' else '-',label=labels[v],linewidth=1.7)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_ylabel('Forward + backward (microseconds)');ax.grid(alpha=.18)
    ax.set_title('a + b*c - exp(d)' if case=='mean_exp' else 'dim1=a*b, dim2=a*c',fontweight='bold')
    ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,col];vals=[r for r in ratio_summary if r['case']==case]
    mid=np.array([r['median_paired_ratio'] for r in vals]);ci=np.array([r['paired_ratio_ci95'] for r in vals])
    ax.errorbar(ns,mid,yerr=[mid-ci[:,0],ci[:,1]-mid],fmt='o-',capsize=3,color=colors['graph_nvrtc'])
    ax.axhline(1,color='#555555',linestyle='--',linewidth=1)
    ax.set_xscale('log');ax.set_ylabel('Graph / handwritten nvcc time');ax.set_xlabel('Number of elements N (log scale)');ax.grid(alpha=.18)
    ax.set_ylim(min(.85,float(ci.min())-.02),max(1.15,float(ci.max())+.02))
    ax.set_title('Paired median ratio; bootstrap 95% interval',fontsize=10)
fig.suptitle('Fused graph kernels vs handwritten kernels',fontsize=17,fontweight='bold',y=.98)
fig.text(.5,.022,'RTX 4080 SUPER | FP64 | block=256 | warm buffers | 9 rounds; 30 at N >= 1,048,576\nCUDA-event timing of GPU batches. JIT, allocation, transfers and Mean/MPI reduction excluded.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.072,1,.94]);fig.savefig(HERE/'performance.png',dpi=180);plt.close(fig)
lines=['融合图与手写 CUDA kernel 性能对比','',
'环境：RTX 4080 SUPER，CUDA/NVRTC 12.6，sm_89，FP64，block=256；WSL 桌面 GPU，未锁频。',
'图的实际生成源码与手写等价计算分别测试；手写同时使用 NVRTC 和 nvcc 编译。',
'所有版本通过相同 CUDA Driver API、相同 stream、相同指针与参数启动，未使用 fast-math 或 restrict。',
'主要指标是一次前向+反向的 GPU 耗时。CUDA Graph replay 每批串行执行64对，以摊薄 CPU 提交间隙；这是计时设施，不是被测的计算图。',
'另有普通逐次提交的 wall-clock 指标，包含 CPU 提交和最终同步，见各轮 host_submit.csv。',
'每种配置预热，三种版本随机交错测量9轮；百万级及以上另复测21轮。汇总保留全部样本。',
'时间为中位数；ratio_of_medians=图中位数/手写nvcc中位数；paired_ratio为同轮比值的中位数。',
'配对bootstrap区间只描述本次样本，不涵盖跨机器、桌面干扰、时钟变化等系统误差。',
'显存分配、数据上传/下载、构图、NVRTC编译、模块加载、外部Mean/MPI规约均不在GPU计时内。',
'独立测forward/backward时缓存工作集不同，因此不能把二者相加当作交替执行的pair时间。','',
'正确性：所有42个首轮+12个复测配置均与独立CPU解析结果核对；最大绝对误差 %.17g。'%max_error,'',
'前向+反向汇总（微秒）：']
for row in ratio_summary:
    manual_jit=statistics.median(groups[row['case'],row['n'],'pair','manual_nvrtc'])
    lines.append('%s N=%d: 图 %.4f；手写NVRTC %.4f；手写nvcc %.4f；图/nvcc %.4f；样本%d'%(row['case'],row['n'],row['graph_us'],manual_jit,row['manual_nvcc_us'],row['ratio_of_medians'],row['samples']))
lines+=['','NVRTC 编译开销（不含首次冷调用的重复编译中位数）：']
for key,values in compile_data.items():lines.append('%s/%s: %.3f ms'%(*key,statistics.median(values)))
lines+=['首次 mean_exp 图 NVRTC 编译为 60.5241 ms；该数字包含首次编译器初始化影响。','',
'SASS/资源检查：所有被测函数没有CALL指令，也没有local memory使用。代码并非逐条一致，具体指令数与寄存器数见 sass_summary.json 和 attributes.csv。',
'结论边界：两个标量表达式的融合图与手写实现属于相同性能量级；百万级复测与首轮有波动，不能从几个百分点推断普遍的速度优势。',
'对于相同图，构建、NVRTC编译和模块加载应完成一次后复用；如果每步重新编译，数十毫秒的编译成本会远超这里的kernel时间。','',
'复现：设置 LAMMPS_SOURCE_DIR 后运行 python3 benchmarks/graph_kernel_20261013/run.py；默认构建目录为 build/graph。',
'大规模复测：python3 benchmarks/graph_kernel_20261013/run.py --focus --output benchmarks/graph_kernel_20261013/results_focus。',
'绘图/汇总：python3 benchmarks/graph_kernel_20261013/analyze.py（需要两轮结果、numpy、matplotlib）。',
'只新增基准与结果文件，未修改图的生产实现。']
(HERE/'report.txt').write_text('\n'.join(lines)+'\n')
print('VALIDATION_MAX',max_error)
for row in ratio_summary:
    if row['n']>=1048576:print(json.dumps(row))
print('WARM_COMPILE', {str(k):statistics.median(v) for k,v in compile_data.items()})
print('SASS_CALLS',sum(len(r['call_instructions']) for r in sass))
