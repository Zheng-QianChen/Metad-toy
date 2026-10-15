#!/usr/bin/env python3
"""保留每次样本，报告中位数和范围，不把 NVRTC 初始化混进 steady-state。"""
import argparse,json,re,statistics
from pathlib import Path

def summary(directory):
    rows=json.loads((directory/'results.json').read_text())
    groups=[]
    for np in sorted({r['np'] for r in rows}):
        stats={}
        for executor in ['LEGACY','GRAPH']:
            samples=[r['us_per_step']/1000 for r in rows if r['np']==np and r['executor']==executor]
            assert len(samples)==3, 'expected three repeats'
            stats[executor]=dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),samples_ms=samples)
        groups.append(dict(np=np,**stats,graph_time_change_percent=100*(stats['GRAPH']['median_ms']/stats['LEGACY']['median_ms']-1)))
    # 匹配轨迹的末端 CV/能量应保持一致；能量非零，且起终 CV 相同（刚体平移）。
    end_errors=[]
    if all(r.get('motion')=='translate' for r in rows):
        for np in sorted({r['np'] for r in rows}):
            for repeat in range(3):
                values=[]
                for executor in ['LEGACY','GRAPH']:
                    row=next(r for r in rows if (r['np'],r['repeat'],r['executor'])==(np,repeat,executor))
                    text=(Path(row['directory'])/'screen.txt').read_text(errors='replace')
                    lines=[list(map(float,line.split())) for line in text.splitlines() if re.match(r'^\s*\d+\s+4000\s+[-0-9.]',line)]
                    values.append(lines[-1][2:])
                    assert abs(values[-1][-1])>1e-6
                err=max(abs(x-y) for x,y in zip(*values));assert err<1e-11
                end_errors.append(err)
    rtc=[r['nvrtc_ms'] for r in rows if r['executor']=='GRAPH']
    return dict(configurations=groups,nvrtc_rank0_ms=dict(min=min(rtc),median=statistics.median(rtc),max=max(rtc)),
                matched_final_cv_energy_max_error=max(end_errors) if end_errors else None,
                samples=rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=summary(args.directory);args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
