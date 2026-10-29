#!/usr/bin/env python3
"""Validate a shared-online journal and export one exact frozen table snapshot."""
import argparse,json,math
from pathlib import Path

def checksum(text):
    value=14695981039346656037
    for byte in text.encode():value=((value^byte)*1099511628211)&((1<<64)-1)
    return f'{value:016x}'

def read(path,step=None):
    with Path(path).open() as source:
        if source.readline()!='# METAD_TOY_SHARED_ONLINE 1\n':raise ValueError('Unsupported shared history header')
        line=source.readline()
        if not line.startswith('CONFIG ') or not line.endswith('\n'):raise ValueError('Missing configuration')
        config=line[len('CONFIG '):-1];fields=config.split()
        if len(fields)!=11:raise ValueError('Invalid configuration')
        nodes=int(fields[0]);lower,upper,sigma,height,thermal,bias_factor=map(float,fields[1:7])
        if nodes<4 or nodes>1000000 or not all(map(math.isfinite,(lower,upper,sigma,height,thermal,bias_factor))) or upper<=lower:raise ValueError('Invalid grid')
        previous=None;chosen=None;count=0
        for line in source:
            if not line.endswith('\n'):raise ValueError('Truncated history record')
            text,hash_value=line[:-1].rsplit(' ',1)
            if checksum(config+'\n'+text)!=hash_value:raise ValueError('History checksum mismatch')
            row=text.split()
            if len(row)!=3+2*nodes or row[0]!='STATE':raise ValueError('Invalid state record')
            current,sequence=int(row[1]),int(row[2]);values=list(map(float,row[3:]))
            if not all(map(math.isfinite,values)):raise ValueError('Nonfinite table')
            if (previous is None and (current,sequence)!=(-1,0)) or (previous is not None and (current<=previous or sequence!=count)):raise ValueError('Invalid history sequence')
            if step is None or current==step:chosen=(current,values)
            previous=current;count+=1
        if chosen is None:raise ValueError('Requested snapshot is absent')
        return dict(nodes=nodes,lower=lower,upper=upper,sigma=sigma,height=height,thermal_energy=thermal,bias_factor=bias_factor,
            well_tempered=fields[7]=='1',aggregation='MEAN' if fields[8]=='0' else 'SUM',
            deposition='MEAN' if fields[9]=='1' else 'SUM',identity=fields[10],depositions=count-1,last_step=previous,
            exported_step=chosen[0],values=chosen[1])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('history',type=Path)
    parser.add_argument('--step',type=int,help='Exact recorded deposition step; default last snapshot')
    parser.add_argument('--output',type=Path,help='New frozen table path; existing files are not overwritten')
    args=parser.parse_args();record=read(args.history,args.step);values=record.pop('values')
    if args.output:
        spacing=(record['upper']-record['lower'])/(record['nodes']-1)
        with args.output.open('x') as target:
            target.write('# METAD_TOY_SHARED_FROZEN 1\n')
            target.write(f"# exported deposition step {record['exported_step']}; aggregation {record['aggregation']}\n")
            for i in range(record['nodes']):target.write(f"{record['lower']+i*spacing:.17g} {values[2*i]:.17g} {values[2*i+1]:.17g}\n")
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
