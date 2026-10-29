"""Independent local verifier. The teacher's verifier was not included in the starter."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from common import RESULTS, ROOT, read_json, write_json, digest, timestamp
from salt import get_student_params
from dense_kernel import inputs
from scaling_study import reference, process_counts
import numpy as np


def tsv(name):
    with (RESULTS/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f,delimiter='\t'))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def verify(smoke=False):
    p=read_json('salted_params.json'); host=read_json('host_profile.json')
    expected=get_student_params(p['student_id'])
    if smoke:
        expected.update(N=64,buffer_elements=65_536)
    require(p==expected,'Salted parameters do not match student ID / run mode.')
    require(host['physical_cores']>=1,'Invalid physical core count.')
    print('[PASS] M0: salted parameters and recorded host topology.')
    stride=tsv('stride_results.tsv')
    require(len(stride)==40,'Expected eight strides x five recorded runs.')
    for s in (1,2,4,8,16,32,64,128):
        rr=[r for r in stride if int(r['stride'])==s]
        require(len(rr)==5,f'Missing stride {s}')
        touched=(p['buffer_elements']+s-1)//s
        require(all(int(r['elements_touched'])==touched and float(r['sum'])==touched and float(r['time_ns'])>0 for r in rr),'Invalid stride reduction or timing')
    n=p['N']; m=n//2; a,b=inputs(n,p['seed']); ref=a[:m,:m]@b[:m,:m]
    matrices=[np.load(RESULTS/f'dense_{kind}.npy',allow_pickle=False) for kind in ('ijk','ikj')]
    metadata=read_json('loop_interchange.json')
    for kind,c in zip(('ijk','ikj'),matrices):
        require(c.shape==(m,m) and np.isfinite(c).all(),'Invalid dense matrix')
        np.testing.assert_allclose(c,ref,rtol=0,atol=1e-12)
        require(digest(c,p['student_id'])==metadata['algorithms'][kind]['digest'],'Stored dense digest mismatch')
    np.testing.assert_allclose(matrices[0],matrices[1],rtol=0,atol=1e-12)
    require(digest(matrices[0],p['student_id'])==digest(matrices[1],p['student_id']),'Dense salted digest mismatch')
    print('[PASS] M1: both loop orders match independent NumPy reference at atol=1e-12; salted digests match.')
    raw=read_json('false_sharing_raw.json')['runs']; fs=tsv('false_sharing_karp_flatt.tsv')
    repeats=1 if smoke else 3; k=20_000 if smoke else 5_000_000
    require(len(raw)==len(fs)==12*repeats,'Missing false-sharing runs')
    for mode in ('unpadded','padded','privatized'):
        for workers in (1,2,4,8):
            group=[r for r in raw if r['mode']==mode and r['P']==workers]
            require(len(group)==repeats,'Missing counter configuration')
            for r in group:
                require(r['K']==k and r['counts']==[k]*workers and r['time_ns']>0,'Incorrect counters/timing')
    for row in fs:
        workers=int(row['P']); mode=row['mode']
        base=next(float(r['median_time_ns']) for r in fs if r['mode']==mode and int(r['P'])==1)
        speed=base/float(row['median_time_ns'])
        if workers>1:
            require(abs(float(row['karp_flatt_literal'])-(workers/speed-1)/(workers-1))<1e-10,'Karp-Flatt mismatch')
    print('[PASS] M2: every worker counter equals K; all totals equal P*K; metric arithmetic verified.')
    scaling=tsv('scaling_metrics.tsv'); sr=read_json('scaling_raw.json')
    expected_counts=process_counts(min(host['physical_cores'],2) if smoke else host['physical_cores'])
    require(sr['tested_process_counts']==expected_counts,'Incomplete physical-core sweep')
    require(sr['repeats']==repeats,'Unexpected scaling repetition count')
    require(len(sr['runs'])==2*len(expected_counts)*repeats,'Missing raw scaling timings')
    for mode in ('strong','weak'):
        for workers in expected_counts:
            samples=[r for r in sr['runs'] if r['mode']==mode and r['P']==workers]
            require(len(samples)==repeats and all(r['time_ns']>0 for r in samples),'Invalid scaling sample set')
    for mode in ('strong','weak'):
        group=[r for r in scaling if r['mode']==mode]
        require(sorted(int(r['P']) for r in group)==expected_counts,'Missing scaling configurations')
        for r in group:
            n=int(r['N']); workers=int(r['P']); steps=int(r['steps'])
            import math
            expected_n=p['N'] if mode=='strong' else math.ceil(p['N']*math.sqrt(workers))
            require(n==expected_n and steps==(3 if smoke else 30),'Incorrect stencil dimensions/iterations')
            parts=[]
            for filename in json.loads(r['output_files']):
                with np.load(RESULTS/filename,allow_pickle=False) as archive:
                    parts.append(archive['values'])
            out=np.concatenate(parts,axis=0)
            ref=reference(n,p['seed'],steps)
            require(out.shape==ref.shape and np.isfinite(out).all(),'Invalid stencil output')
            np.testing.assert_allclose(out,ref,rtol=0,atol=1e-12)
            require(digest(out,p['student_id'])==digest(ref,p['student_id'])==r['sha256'],'Stencil salted digest mismatch')
    print('[PASS] M3: all strong/weak outputs independently recomputed; tolerance and SHA-256 verified.')
    roof=read_json('roofline_summary.json')
    warnings=list(host.get('warnings',[]))
    if roof['peak_theoretical_bandwidth_gbs'] is None or roof['peak_theoretical_fp64_gflops'] is None:
        warnings.append('Roofline hardware peak ceilings are unavailable: quantitative classification remains undetermined.')
    fs8={mode:next(float(r['median_time_ns']) for r in fs if r['mode']==mode and int(r['P'])==8) for mode in ('unpadded','padded')}
    fs1=next(float(r['median_time_ns']) for r in fs if r['mode']=='unpadded' and int(r['P'])==1)
    if fs8['unpadded']<=fs1 or fs8['unpadded']<=fs8['padded']:
        warnings.append('Required false-sharing collapse/recovery is not clear in these measurements. Inspect placement and rerun honestly; do not alter data.')
    stride_medians={s:float(next(r for r in stride if int(r['stride'])==s)['median_ns_per_element']) for s in (1,8)}
    if stride_medians[8]<=stride_medians[1]:
        warnings.append('Stride-8 cliff not observed; inspect backend, cache capacity and CPU load.')
    files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(RESULTS.iterdir())
           if f.is_file() and f.name not in ('verification_receipt.sig','smoke_receipt.json','.gitkeep') and not f.name.endswith('.tmp')}
    source_files=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'src').glob('*.c')),ROOT/'verify_outputs.py',ROOT/'run_lab.py']
    for f in source_files:
        files[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    receipt={'created_utc':timestamp(),'student_id':p['student_id'],'numerical_checks':'PASS','smoke_only':smoke,
        'rubric_readiness':'REVIEW_WARNINGS' if warnings else 'NUMERICAL_AND_METADATA_CHECKS_PASS',
        'warnings':warnings,'sha256_files':files,
        'verifier':'Independent local verifier; not the absent instructor-supplied verify_outputs.py.',
        'signature_note':'Unsigned integrity manifest, not proof of physical hardware, authorship or teacher acceptance.'}
    receipt['manifest_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json('smoke_receipt.json' if smoke else 'verification_receipt.sig',receipt)
    for warning in warnings:
        print('[REVIEW]',warning)
    print('Numerical checks passed. Review metadata/pathology warnings before submitting.')
    return receipt


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--smoke',action='store_true'); args=a.parse_args()
    if args.smoke and os.environ.get('LAB_SMOKE')!='1':
        raise SystemExit('Use run_lab.py --smoke; formal and smoke outputs must stay separate.')
    verify(args.smoke)
