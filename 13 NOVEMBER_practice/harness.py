from dataclasses import dataclass
import multiprocessing as mp
import time

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float

def fnv1a_hash(s):
    h = 14695981039346656037
    for b in s.encode('utf-8'):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def run_benchmark(fn, warmup=2, runs=7):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns()-t)/1e6)
    samples.sort()
    return BenchStats(samples[0], samples[runs//2], samples[3*runs//4]-samples[runs//4])

def emit_verify_tag(phase, seed, m1, m2):
    hw = mp.cpu_count()
    sig = fnv1a_hash(f'{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}')
    return f'VERIFY_HASH[{phase}]: {sig:016X} (P_log={hw})'
