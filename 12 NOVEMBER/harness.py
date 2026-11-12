from dataclasses import dataclass
import multiprocessing as mp
import time

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float

def fnv1a_hash(s):
    h = 14695981039346656037
    for b in s.encode('utf-8'):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def summarize(samples):
    a = sorted(samples)
    n = len(a)
    med = a[n//2] if n % 2 else (a[n//2-1] + a[n//2]) / 2
    return BenchStats(a[0], med, a[3*n//4] - a[n//4], a[-1])

def run_benchmark(fn, warmup_runs=3, timed_runs=9):
    for _ in range(warmup_runs):
        fn()
    a = []
    for _ in range(timed_runs):
        t = time.perf_counter_ns()
        fn()
        a.append((time.perf_counter_ns() - t) / 1e6)
    return summarize(a)

def emit_verify_tag(task_id, student_seed, primary_metric, secondary_metric):
    hw = mp.cpu_count()
    payload = f'{task_id}|{student_seed}|{hw}|{primary_metric:.4f}|{secondary_metric:.4f}'
    return f'VERIFY_HASH[{task_id}]: {fnv1a_hash(payload):016X} (P_log={hw})'
