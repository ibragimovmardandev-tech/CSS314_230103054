"""AI-assisted reference implementation. Standard library only, Python >=3.8.
Run on your own laptop; --smoke is a correctness check, not submission data.
"""
import argparse
import ctypes
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from harness import summarize, emit_verify_tag

ROOT = Path(__file__).resolve().parent

class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file
    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return 'Unavailable: ' + str(e)

def topology():
    h = dict(model=platform.processor(), physical=None, logical=mp.cpu_count(),
             line=None, l1=None, l2=None, l3=None, os=platform.platform(),
             python=platform.python_version(), physical_cpus=[], logical_cpus=[], raw='')
    if platform.system() == 'Linux':
        h['raw'] = command(['lscpu'])
        allowed = sorted(os.sched_getaffinity(0))
        groups = {}
        for c in allowed:
            base = Path(f'/sys/devices/system/cpu/cpu{c}')
            try:
                key = tuple((base/'topology'/v).read_text().strip()
                            for v in ['physical_package_id', 'core_id'])
                groups.setdefault(key, []).append(c)
            except OSError:
                pass
        if groups:
            h['physical_cpus'] = [v[0] for v in groups.values()]
            h['logical_cpus'] = h['physical_cpus'] + [c for c in allowed if c not in h['physical_cpus']]
            h['physical'] = len(groups)
        for line in h['raw'].splitlines():
            if line.startswith('Model name:'):
                h['model'] = line.split(':', 1)[1].strip()
        for d in Path(f'/sys/devices/system/cpu/cpu{allowed[0]}/cache').glob('index*'):
            level = (d/'level').read_text().strip()
            typ = (d/'type').read_text().strip()
            if typ != 'Instruction':
                h['l'+level] = (d/'size').read_text().strip()
                h['line'] = int((d/'coherency_line_size').read_text())
        h['allowed_cpus'] = allowed
    elif platform.system() == 'Darwin':
        keys = dict(model='machdep.cpu.brand_string', physical='hw.physicalcpu',
                    line='hw.cachelinesize', l1='hw.l1dcachesize', l2='hw.l2cachesize', l3='hw.l3cachesize')
        for field, key in keys.items():
            val = command(['sysctl', '-n', key])
            h[field] = int(val) if val.isdigit() else val
        h['raw'] = json.dumps(h, indent=2)
    elif platform.system() == 'Windows':
        raw = command(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | ConvertTo-Json'])
        h['raw'] = raw
        try:
            rows = json.loads(raw)
            rows = rows if isinstance(rows, list) else [rows]
            h.update(model='; '.join(r['Name'] for r in rows),
                     physical=sum(r['NumberOfCores'] for r in rows),
                     l2=str(rows[0]['L2CacheSize'])+' KB', l3=str(rows[0]['L3CacheSize'])+' KB')
        except (ValueError, KeyError):
            pass
        # Native Windows topology: core masks and actual cache descriptors.
        class Cache(ctypes.Structure):
            _fields_=[('level',ctypes.c_ubyte),('associativity',ctypes.c_ubyte),
                      ('line',ctypes.c_ushort),('size',ctypes.c_ulong),('kind',ctypes.c_int)]
        class InfoUnion(ctypes.Union):
            _fields_=[('cache',Cache),('reserved',ctypes.c_ulonglong*2)]
        class Info(ctypes.Structure):
            _fields_=[('mask',ctypes.c_size_t),('relationship',ctypes.c_int),('data',InfoUnion)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        getinfo=kernel.GetLogicalProcessorInformation
        getinfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        getinfo.restype=ctypes.c_int
        length=ctypes.c_ulong(0)
        getinfo(None,ctypes.byref(length))
        if length.value:
            buf=ctypes.create_string_buffer(length.value)
            if getinfo(buf,ctypes.byref(length)):
                entries=ctypes.cast(buf,ctypes.POINTER(Info))
                cores=[]
                for i in range(length.value//ctypes.sizeof(Info)):
                    entry=entries[i]
                    if entry.relationship==0:
                        cores.append([c for c in range(64) if entry.mask & (1<<c)])
                    elif entry.relationship==2:
                        cache=entry.data.cache
                        if cache.kind in (0,2):  # Unified or data cache.
                            h['l'+str(cache.level)]=str(cache.size)+' bytes'
                            h['line']=cache.line
                if cores and h['logical']<=64:
                    h['physical_cpus']=[c[0] for c in cores]
                    h['logical_cpus']=h['physical_cpus']+[c for core in cores for c in core[1:]]
                    h['physical']=len(cores)
    return h

def pin(cpu):
    if cpu is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})
    elif cpu is not None and platform.system()=='Windows':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        kernel.SetProcessAffinityMask.restype=ctypes.c_int
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1<<cpu):
            raise ctypes.WinError(ctypes.get_last_error())

def bounds(n, p, t):
    return n*t//p, n*(t+1)//p

def process_entry(fn, args, ready, start, cpu):
    pin(cpu)
    ready.release()
    start.wait()
    fn(*args)

def processes(fn, args, cpus):
    """Creation excluded; release-to-join timing, including completion overhead."""
    start, ready = mp.Event(), mp.Semaphore(0)
    workers = [mp.Process(target=process_entry,
               args=(fn, a, ready, start, cpus[t] if t < len(cpus) else None))
               for t, a in enumerate(args)]
    try:
        for w in workers:
            w.start()
        for _ in workers:
            if not ready.acquire(timeout=120):
                raise RuntimeError('Worker initialization timed out')
        t0 = time.perf_counter_ns()
        start.set()
        for w in workers:
            w.join()
        elapsed = (time.perf_counter_ns()-t0)/1e6
        if any(w.exitcode != 0 for w in workers):
            raise RuntimeError('Worker failure; results discarded')
        return elapsed
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join()

def measure(label, fn, warm, repeats):
    vals, details = [], []
    for trial in range(warm+repeats):
        result = fn()
        ms, extra = result if isinstance(result, tuple) else (result, {})
        print(f'{label} trial={trial+1} warmup={trial<warm} ms={ms:.6f}', flush=True)
        if trial >= warm:
            vals.append(ms)
            details.append(extra)
    return dict(label=label, **vars(summarize(vals)), samples_ms=vals, details=details)

def fs_worker(name, slot, iters):
    shm = shared_memory.SharedMemory(name=name)
    arr = (ctypes.c_uint64*(shm.size//8)).from_buffer(shm.buf)
    acc = arr[slot]
    for i in range(iters):
        acc = (acc*6364136223846793005 + (i|1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot] = acc
    del arr
    shm.close()

def fs_trial(p, stride, n, cpus):
    # 4096 bytes are insufficient when (p-1)*stride >= 512; extend safely.
    shm = shared_memory.SharedMemory(create=True, size=max(4096, ((p-1)*stride+1)*8))
    try:
        shm.buf[:] = bytes(shm.size)
        args = [(shm.name, t*stride, bounds(n,p,t)[1]-bounds(n,p,t)[0]) for t in range(p)]
        return processes(fs_worker, args, cpus)
    finally:
        shm.close()
        shm.unlink()

def compute_block(pair):
    lo, hi = pair
    total = 0.0
    for i in range(lo, hi):
        x = (i % 997 + 1)*0.001
        for _ in range(35):
            x = x*0.999999 + 0.000001 + math.sin(x)*0.001
        total += x
    return total

def serial_stage(n):
    acc = 0.0
    kept = 0
    for i in range(n//20):
        x = (i % 997+1)*0.001
        for _ in range(6):  # Exactly 12 transcendental calls per element.
            x = math.sqrt(abs(math.sin(x))+0.01)
        acc += x
        kept += acc > 0.0  # Deterministic prefix filter.
    return acc, kept

def pool_init(cpus, counter):
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    pin(cpus[rank] if rank < len(cpus) else None)

def scaling_trial(pool, p, n):
    t0 = time.perf_counter_ns()
    a = serial_stage(n)
    ta = time.perf_counter_ns()
    result = pool.map(compute_block, [bounds(n,p,t) for t in range(p)], chunksize=1)
    end = time.perf_counter_ns()
    return (end-t0)/1e6, dict(serial_ms=(ta-t0)/1e6, checksum=a[0]+sum(result))

def bandwidth_worker(a, b, lo, hi):
    sa, sb = shared_memory.SharedMemory(name=a), shared_memory.SharedMemory(name=b)
    sa.buf[lo:hi] = sb.buf[lo:hi]
    sa.close()
    sb.close()

def fma_worker(loops, out, rank):
    a,b = 0.9999999,0.0000001
    x0=x1=x2=x3=x4=x5=x6=x7=0.5
    x8=x9=x10=x11=x12=x13=x14=x15=0.5
    for _ in range(loops):
        x0=x0*a+b; x1=x1*a+b; x2=x2*a+b; x3=x3*a+b
        x4=x4*a+b; x5=x5*a+b; x6=x6*a+b; x7=x7*a+b
        x8=x8*a+b; x9=x9*a+b; x10=x10*a+b; x11=x11*a+b
        x12=x12*a+b; x13=x13*a+b; x14=x14*a+b; x15=x15*a+b
    out[rank] = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15

def gemm_worker(names, n, lo, hi, variant, bs):
    shms = [shared_memory.SharedMemory(name=s) for s in names]
    a,b,c = [s.buf.cast('d') for s in shms]
    if variant == 1:
        for i in range(lo, hi):
            row=i*n
            for j in range(n):
                dot=0.0
                for k in range(n):
                    dot += a[row+k]*b[k*n+j]
                c[row+j]=dot
    elif variant == 2:
        for i in range(lo, hi):
            row=i*n
            for k in range(n):
                aik=a[row+k]
                bk=k*n
                for j in range(n):
                    c[row+j] += aik*b[bk+j]
    else:
        for ii in range(lo, hi, bs):
            for jj in range(0,n,bs):
                je=min(jj+bs,n)
                for kk in range(0,n,bs):
                    for i in range(ii,min(ii+bs,hi)):
                        cv=c[i*n+jj:i*n+je]
                        for k in range(kk,min(kk+bs,n)):
                            aik=a[i*n+k]
                            bv=b[k*n+jj:k*n+je]
                            for j in range(je-jj):
                                cv[j] += aik*bv[j]
                            bv.release()
                        cv.release()
    a.release(); b.release(); c.release()
    for s in shms:
        s.close()

def schedule_worker(t, p, n, strategy, chunk, counter, times, sums, counts):
    begin=time.perf_counter_ns()
    checksum=0.0
    count=0
    def work(i):
        x=0.001*(i+1)
        for _ in range(1+(i*i)//3600):
            x=math.sqrt(abs(math.sin(x)+math.cos(x))+0.001)
        return x
    if strategy == 'block':
        items=range(*bounds(n,p,t))
    elif strategy == 'cyclic':
        items=range(t,n,p)
    else:
        items=None
    if items is not None:
        for i in items:
            checksum+=work(i)
            count+=1
    else:
        while True:
            with counter.get_lock():
                lo=counter.value
                counter.value+=chunk
            if lo>=n:
                break
            for i in range(lo,min(lo+chunk,n)):
                checksum+=work(i)
                count+=1
    times[t]=(time.perf_counter_ns()-begin)/1e6
    sums[t]=checksum
    counts[t]=count

def schedule_trial(p,n,strategy,chunk,cpus):
    times,sums=mp.Array('d',p,lock=False),mp.Array('d',p,lock=False)
    counts=mp.Array('i',p,lock=False)
    counter=mp.Value('i',0)
    args=[(t,p,n,strategy,chunk,counter,times,sums,counts) for t in range(p)]
    ms=processes(schedule_worker,args,cpus)
    assert sum(counts)==n, 'Lost or duplicated scheduling items'
    return ms,dict(worker_ms=list(times), imbalance=max(times)/statistics.mean(times),
                   checksum=sum(sums), items=list(counts))

def table(headers, rows):
    def val(x):
        return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(val,r))+' |' for r in rows])+'\n'

def report(state, path):
    h=state['hardware']
    text=['# Parallel Computing Lab - '+str(state['student_id']),
          '**AI-assisted reference implementation.** Measurements belong to the machine running this program.',
          'Mode: '+state['mode']+'. Smoke results are not submission measurements.',
          'CPU affinity: '+('one worker per logical CPU, physical cores first' if h['logical_cpus'] else 'UNPINNED: OS scheduling; strict physical-core placement not established'),
          'Timing: process creation is outside each trial; release, shared-memory attachment, kernel and join are included. Phase 2 uses a persistent Pool. Full trials use 3 warmups/9 samples, except Phase 1 (2/7). No elapsed time is fabricated.',
          '## Phase 0: Hardware baseline', table(['Parameter','Value'],[(k,v) for k,v in h.items() if k!='raw']),
          '```text\n'+h['raw']+'\n```',
          f"Clock-read loop latency: {state['clock_ns']:.6f} ns (includes loop overhead).",
          f"Timer minimum positive observed delta: {state['timer_step_ns']} ns."]
    for tag in state.get('tags',[]):
        text.append('```text\n'+tag+'\n```')
    if 'stride' in state:
        a=state['stride']
        text += ['## Phase 1: False sharing', '### Table 1A',
                 table(['Stride','Bytes','Median ms','IQR ms','Mops/s'],
                 [(r['stride'],r['stride']*8,r['median_ms'],r['iqr_ms'],state['m']/r['median_ms']/1000) for r in a])]
        drops=[(a[i-1]['median_ms']-a[i]['median_ms'])/a[i-1]['median_ms'] for i in range(1,len(a))]
        j=max(range(len(drops)),key=drops.__getitem__)
        text += [f'Largest adjacent improvement: {drops[j]*100:.3f}% at {a[j+1]["stride"]*8} bytes. '
                 'This is a candidate transition, not proof of cache-line width; compare OS line size and IQR. Python dispatch and scheduling can obscure false sharing.']
        rows=[]
        for stride in (1,16):
            records=[r for r in state['fs_scale'] if r['stride']==stride]
            t1=records[0]['median_ms']
            for r in records:
                p=r['p']; speed=t1/r['median_ms']
                rows.append([stride,p,r['median_ms'],r['iqr_ms'],speed,speed/p,
                             (1/speed-1/p)/(1-1/p) if p>1 else 'N/A'])
        text += ['### Table 1B',table(['Stride','p','ms','IQR','Speedup','Efficiency','e(p)'],rows),
                 'Karp-Flatt includes communication, imbalance, process completion and interpreter overhead, not only serial code. Rising e(p) at stride 1 is consistent with coherence contention only if the isolated series and noise support that interpretation.']
        for stride in (1,16):
            series=[r for r in rows if r[0]==stride and r[1]>1]
            if series:
                text.append(f'Stride {stride}: e(p) changes from {series[0][-1]:.6f} (p={series[0][1]}) to {series[-1][-1]:.6f} (p={series[-1][1]}).')
    if 'strong' in state:
        srows,wrows=[],[]
        t1=state['strong'][0]['median_ms']; w1=state['weak'][0]['median_ms']
        for r in state['strong']:
            speed=t1/r['median_ms']
            srows.append([r['p'],r['median_ms'],r['iqr_ms'],speed,speed/r['p']])
        for r in state['weak']:
            alpha=statistics.median([d['serial_ms']/ms for d,ms in zip(r['details'],r['samples_ms'])])
            wrows.append([r['p'],r['n'],r['median_ms'],w1/r['median_ms'],alpha,r['p']-alpha*(r['p']-1)])
        s=statistics.median([d['serial_ms']/ms for d,ms in zip(state['strong'][0]['details'],state['strong'][0]['samples_ms'])])
        p=state['strong'][-1]['p']
        # Least-squares fit T(p)/T(1)=s+(1-s)/p, constrained to [0,1].
        xs=[1-1/r['p'] for r in state['strong']]
        ys=[r['median_ms']/t1-1/r['p'] for r in state['strong']]
        fitted=max(0,min(1,sum(x*y for x,y in zip(xs,ys))/sum(x*x for x in xs))) if p>1 else s
        text += ['## Phase 2: Strong and weak scaling',table(['p','Strong ms','IQR','S','E'],srows),
                 table(['p','N','Weak ms','Weak E','alpha','Gustafson S'],wrows),
                 f'Measured s={s:.6f}; Amdahl limit={1/s:.6f}; predicted S({p})={1/(s+(1-s)/p):.6f}; observed={srows[-1][3]:.6f}. Least-squares fitted serial fraction={fitted:.6f}.',
                 'The serial stage also grows with N in the weak experiment, so ideal constant weak runtime is not expected. Gustafson speedup uses each scaled run\'s measured serial fraction; it is a model, not a measured serial run of the scaled problem.']
        q=h['physical']
        if q>1 and q<p:
            before=srows[q-1][3]-srows[q-2][3]
            after=srows[q][3]-srows[q-1][3]
            drop=100*(before-after)/abs(before) if before else None
            text.append(f'Marginal speedup before physical-core boundary: {before:.6f}; after: {after:.6f}; relative drop={drop}. Negative values mean improvement. Only pinned, verified topology supports attributing this to SMT; hybrid cores need OS topology inspection.')
        else:
            text.append('No measurable physical-to-SMT boundary in this sweep.')
    if 'gemm' in state:
        bw=state['bandwidth_gbs']; peak=state['compute_gflops']
        best=min([r for r in state['gemm'] if r['variant']==3],key=lambda r:r['median_ms'])
        rows=[]
        for r in state['gemm']:
            intensity=r['bs']/12 if r['variant']==3 else 0.125
            roof=min(peak,bw*intensity)
            rows.append([r['variant'],r['bs'] or '-',r['median_ms'],r['iqr_ms'],r['gflops'],
                         24*r['bs']**2 if r['bs'] else '-',r['max_error'],intensity,100*r['gflops']/roof])
        text += ['## Phase 3: Empirical roofline',
                 f'Bandwidth={bw:.6f} GB/s; Python arithmetic ceiling={peak:.6f} GFLOP/s; ridge={peak/bw:.6f} FLOP/byte.',
                 table(['Variant','BS','ms','IQR','GFLOP/s','Tile bytes','Max error','Model I','% model roof'],rows),
                 f'Best BS={best["bs"]}; tile footprint={24*best["bs"]**2} bytes. OS L1d={h["l1"]}; OS L2={h["l2"]}. Compare units carefully (Linux K means KiB). Smaller tiles add Python loop and slice overhead; cache fit alone does not guarantee an optimum.',
                 'Variant 2 intensity is conservatively modeled at 0.125, since exact cache reuse/DRAM traffic is unmeasured. Variant 3 uses BS/12 as requested. All ceilings are empirical Python-level models: x*a+b is two operations, not a guaranteed hardware fused FMA. The 2-byte/read-write bandwidth formula is nominal traffic, excluding write allocation. Warm caches can invalidate a DRAM interpretation, especially if L3 exceeds 192 MB. Cache claims in the handout are hypotheses, not guaranteed measurements.']
    if 'schedule' in state:
        rows=[]
        for r in state['schedule']:
            idx=min(range(len(r['samples_ms'])),key=lambda k:abs(r['samples_ms'][k]-r['median_ms']))
            d=r['details'][idx]
            rows.append([r['strategy'],r['chunk'] or '-',r['median_ms'],r['iqr_ms'],d['imbalance'],
                         ', '.join(f'{v:.3f}' for v in d['worker_ms'])])
        best=min([r for r in state['schedule'] if r['strategy']=='dynamic'],key=lambda r:r['median_ms'])
        text += ['## Phase 4: Scheduling',table(['Strategy','Chunk','Wall ms','IQR','lambda','Worker ms (median wall trial)'],rows),
                 f'Fastest measured dynamic chunk C*={best["chunk"]}. Compare IQR before treating it as a stable optimum. Block assigns the heaviest tail to one worker; cyclic distributes it. C=1 adds lock contention; C=1024 increases tail imbalance. This is centralized dynamic scheduling, not decentralized work stealing. Active times include lock waits and exclude the initial start wait.']
    elapsed=state.get('elapsed_seconds',0)
    text += ['## Run status',f'Completed phases: {state.get("completed",[])}; elapsed seconds: {elapsed:.3f}.',
             'The lab requires at least 120 minutes of genuine work. This tool neither enforces nor certifies that requirement, independent work, or offline execution.',
             '## Raw telemetry', '```json\n'+json.dumps(state,indent=2)+'\n```', '## Python source appendix']
    logfile=ROOT/state['logfile']
    if logfile.exists():
        text += ['## Verbatim terminal telemetry','```text\n'+logfile.read_text(encoding='utf-8')+'\n```']
    for source in sorted(ROOT.glob('*.py')):
        text += ['### '+source.name, '```python\n'+source.read_text(encoding='utf-8')+'\n```']
    path.write_text('\n\n'.join(text)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',type=int,default=230103054)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--physical',type=int,help='Override OS physical core detection')
    parser.add_argument('--line',type=int,help='Manually verified cache line bytes')
    parser.add_argument('--l1',help='Manually verified L1 data cache, include units')
    parser.add_argument('--resume',action='store_true',help='Resume completed phases from JSON')
    args=parser.parse_args()
    if args.id<=0:
        parser.error('Student ID must be positive')
    h=topology()
    if args.physical: h['physical']=args.physical
    if args.line: h['line']=args.line
    if args.l1: h['l1']=args.l1
    if not isinstance(h['physical'],int) or not 1<=h['physical']<=h['logical']:
        parser.error('Cannot determine physical cores; provide --physical N from OS')
    pphys=min(2,h['physical']) if args.smoke else h['physical']
    plog=min(2,h['logical']) if args.smoke else h['logical']
    if not args.smoke and 'allowed_cpus' in h and len(h['allowed_cpus'])<h['logical']:
        parser.error('CPU affinity restricted: full laptop sweep unavailable. Run locally without restriction.')
    cpus=h['logical_cpus']
    suffix='_SMOKE' if args.smoke else ''
    logfile=f'terminal_{args.id}{suffix}.log'
    logstream=(ROOT/logfile).open('a' if args.resume else 'w',encoding='utf-8',buffering=1)
    sys.stdout=Tee(sys.stdout,logstream)
    dest=ROOT/f'lab_report_{args.id}{suffix}.md'
    data=ROOT/f'telemetry_{args.id}{suffix}.json'
    warm,reps=(0,1) if args.smoke else (3,9)
    state=dict(hardware=h,mode='SMOKE' if args.smoke else 'FULL',student_id=args.id,tags=[],completed=[],logfile=logfile)
    if args.resume and data.exists():
        state=json.loads(data.read_text())
        if state['hardware']!=h or state['student_id']!=args.id:
            parser.error('Resume hardware/configuration differs; use a fresh run')
    started=time.monotonic(); previous=state.get('elapsed_seconds',0)
    def save(phase):
        state['completed'].append(phase)
        state['elapsed_seconds']=previous+time.monotonic()-started
        data.write_text(json.dumps(state,indent=2),encoding='utf-8')
        report(state,dest)
        print(f'Saved {dest}',flush=True)
    def tag(phase,a,b):
        line=emit_verify_tag('PHASE'+str(phase),args.id,a,b)
        print(line,flush=True)
        state['tags'].append(line)
    if 0 not in state['completed']:
        n=20000 if args.smoke else 2000000
        start=time.perf_counter_ns()
        for _ in range(n): time.perf_counter_ns()
        state['clock_ns']=(time.perf_counter_ns()-start)/n
        ticks=[time.perf_counter_ns() for _ in range(10000)]
        state['timer_step_ns']=min(b-a for a,b in zip(ticks,ticks[1:]) if b>a)
        tag(0,state['clock_ns'],float(h['logical'])); save(0)
    if 1 not in state['completed']:
        n=20000 if args.smoke else 24000000
        state['m']=n; state['stride']=[]; state['fs_scale']=[]
        fw,fr=(0,1) if args.smoke else (2,7)
        for stride in [1,2,4,8,16,32]:
            r=measure(f'Phase1 p={pphys} stride={stride}',lambda:fs_trial(pphys,stride,n,cpus),fw,fr)
            r.update(stride=stride,p=pphys); state['stride'].append(r)
        ps={1,plog}; q=2
        while q<=plog: ps.add(q); q*=2
        for stride in [1,16]:
            for p in sorted(ps):
                r=measure(f'Phase1 scale p={p} stride={stride}',lambda:fs_trial(p,stride,n,cpus),fw,fr)
                r.update(stride=stride,p=p); state['fs_scale'].append(r)
        tag(1,state['stride'][0]['median_ms'],state['stride'][4]['median_ms']); save(1)
    if 2 not in state['completed']:
        state['strong']=[]; state['weak']=[]
        for p in range(1,plog+1):
            counter=mp.Value('i',0)
            with mp.Pool(p,initializer=pool_init,initargs=(cpus,counter)) as pool:
                pool.map(compute_block,[(0,1)]*(p*2))
                for kind,n in [('strong',2000 if args.smoke else 1600000),('weak',(500 if args.smoke else 250000)*p)]:
                    r=measure(f'Phase2 {kind} p={p}',lambda:scaling_trial(pool,p,n),warm,reps)
                    r.update(p=p,n=n); state[kind].append(r)
        r=state['weak'][-1]
        alpha=statistics.median([d['serial_ms']/ms for d,ms in zip(r['details'],r['samples_ms'])])
        tag(2,state['strong'][0]['median_ms']/state['strong'][-1]['median_ms'],plog-alpha*(plog-1)); save(2)
    if 3 not in state['completed']:
        size=1000000 if args.smoke else 96000000
        a=b=None
        try:
            a=shared_memory.SharedMemory(create=True,size=size)
            b=shared_memory.SharedMemory(create=True,size=size)
            # Touch every page before timing. Avoid a 96MB temporary allocation.
            for offset in range(0,size,1000000):
                length=min(1000000,size-offset)
                a.buf[offset:offset+length]=bytes(length)
                b.buf[offset:offset+length]=b'\x01'*length
            r=measure('Phase3 bandwidth',lambda:processes(bandwidth_worker,
                [(a.name,b.name,*bounds(size,pphys,t)) for t in range(pphys)],cpus),warm,reps)
            assert a.buf[:]==b.buf[:]
            state['bandwidth']=r; state['bandwidth_gbs']=2*size/(r['median_ms']*1e6)
        finally:
            for s in (a,b):
                if s is not None: s.close(); s.unlink()
        loops=2500 if args.smoke else 2500000
        out=mp.Array('d',pphys,lock=False)
        r=measure('Phase3 compute',lambda:processes(fma_worker,[(loops,out,t) for t in range(pphys)],cpus),warm,reps)
        state['compute']=r; state['compute_gflops']=loops*32*pphys/(r['median_ms']*1e6)
        n=17 if args.smoke else 384
        shms=[]; views=[]; state['gemm']=[]
        try:
            for _ in range(3): shms.append(shared_memory.SharedMemory(create=True,size=n*n*8))
            views=[s.buf.cast('d') for s in shms]
            for i in range(n*n):
                views[0][i]=((i*17+3)%101-50)/50
                views[1][i]=((i*13+7)%97-48)/48
            reference=None
            for variant,bs in [(2,0),(1,0)]+[(3,s) for s in [4,8,12,16,24,32,48,64,96,128]]:
                def trial():
                    shms[2].buf[:]=bytes(n*n*8)
                    return processes(gemm_worker,[(tuple(s.name for s in shms),n,*bounds(n,pphys,t),variant,bs) for t in range(pphys)],cpus)
                r=measure(f'Phase3 GEMM v={variant} BS={bs}',trial,warm,reps)
                if reference is None: reference=list(views[2])
                error=max(abs(x-y) for x,y in zip(reference,views[2]))
                if error>=1e-9: raise ArithmeticError(f'GEMM error: {error}')
                r.update(variant=variant,bs=bs,max_error=error,gflops=2*n**3/(r['median_ms']*1e6))
                state['gemm'].append(r)
        finally:
            for v in views: v.release()
            for s in shms: s.close(); s.unlink()
        best=max([r for r in state['gemm'] if r['variant']==3],key=lambda r:r['gflops'])
        naive=next(r for r in state['gemm'] if r['variant']==1)
        tag(3,naive['gflops'],best['gflops']); save(3)
    if 4 not in state['completed']:
        state['schedule']=[]; baseline=None
        for strategy,chunk in [('block',0),('cyclic',0)]+[('dynamic',c) for c in [1,4,16,64,256,1024]]:
            r=measure(f'Phase4 {strategy} C={chunk}',lambda:schedule_trial(pphys,180 if args.smoke else 12000,strategy,chunk,cpus),warm,reps)
            for d in r['details']:
                if baseline is None: baseline=d['checksum']
                if not math.isclose(baseline,d['checksum'],rel_tol=1e-10,abs_tol=1e-9):
                    raise ArithmeticError('Scheduling checksum mismatch')
            r.update(strategy=strategy,chunk=chunk); state['schedule'].append(r)
        best=min(state['schedule'][2:],key=lambda r:r['median_ms'])
        tag(4,state['schedule'][0]['median_ms'],best['median_ms']); save(4)
    print('DONE:',dest)

if __name__=='__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
