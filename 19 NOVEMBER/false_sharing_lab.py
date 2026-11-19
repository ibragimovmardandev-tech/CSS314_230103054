"""Run with Python 3, NumPy and Numba. Outputs go beside this script."""
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import llvmlite
import numba
import numpy as np
from llvmlite import binding, ir
from numba import njit, types
from numba.extending import intrinsic

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "ibragimovmardandev-tech"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
REPEATS = 3
MASK = (1 << 64) - 1
CASES = ("stride_1", "stride_4", "stride_8", "stride_16", "private")
OUT = Path(__file__).resolve().parent


def detect_cache_line_size():
    try:
        path = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
        return int(path.read_text().strip()), "sysfs"
    except (OSError, ValueError):
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"])), "sysctl"
        return 64, "specification fallback"


def init_system_context(git_user):
    cpuinfo = Path("/proc/cpuinfo")
    records = []
    if cpuinfo.exists():
        for block in cpuinfo.read_text().strip().split("\n\n"):
            records.append(dict(tuple(part.strip() for part in line.split(":", 1))
                                for line in block.splitlines() if ":" in line))
    model = (records[0].get("model name") if records else None) or platform.processor() or platform.machine()
    if not git_user.strip() or git_user.startswith("REPLACE_"):
        raise ValueError("Set the Git account handle.")
    logical = os.cpu_count() or 1
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(logical))
    if len(allowed) < 4:
        raise RuntimeError("Four available logical CPUs are required.")
    by_cpu = {int(r["processor"]): r for r in records if r.get("processor","").isdigit()}
    selected, seen = [], set()
    for cpu in allowed:
        r = by_cpu.get(cpu, {})
        if "physical id" in r and "core id" in r:
            key = (r["physical id"], r["core id"])
            if key not in seen:
                selected.append(cpu)
                seen.add(key)
    distinct = len(selected)
    selected.extend(i for i in allowed if i not in selected)
    line, source = detect_cache_line_size()
    digest = hashlib.sha256(f"{git_user}::{model}".encode()).hexdigest()
    return dict(git_username=git_user, student_id="230103054", cpu_model=model,
                logical_cores=logical, allowed_cpus=allowed, selected_cpus=selected[:4],
                distinct_reported_cores=distinct, pinning=hasattr(os, "sched_setaffinity"),
                cache_line_bytes=line, cache_line_source=source, instance_seed=int(digest[:16],16),
                python=platform.python_version(), os=platform.system(), numpy=np.__version__,
                numba=numba.__version__, llvmlite=llvmlite.__version__)


# A recurrence alone permits register promotion. Mark the slot accesses volatile
# in native IR. Each worker owns its slot, so atomic arithmetic is unnecessary.
class VolatileLoad(ir.instructions.LoadInstr):
    def descr(self, buf):
        super().descr(buf)
        buf[-1] = buf[-1].replace("load ", "load volatile ", 1)


class VolatileStore(ir.instructions.StoreInstr):
    def descr(self, buf):
        super().descr(buf)
        buf[-1] = buf[-1].replace("store ", "store volatile ", 1)


def slot_address(context, builder, arr_type, array, slot):
    data = context.make_array(arr_type)(context, builder, value=array).data
    return builder.gep(data, [slot])


@intrinsic
def read_slot(typingctx, array, slot):
    def codegen(context, builder, sig, args):
        ptr = slot_address(context, builder, sig.args[0], *args)
        inst = VolatileLoad(builder.block, ptr, typ=ir.IntType(64))
        inst.align = 8
        builder._insert(inst)
        return inst
    return types.uint64(array, slot), codegen


@intrinsic
def write_slot(typingctx, array, slot, value):
    def codegen(context, builder, sig, args):
        ptr = slot_address(context, builder, sig.args[0], args[0], args[1])
        inst = VolatileStore(builder.block, args[2], ptr)
        inst.align = 8
        builder._insert(inst)
        return context.get_dummy_value()
    return types.void(array, slot, value), codegen


@njit(inline="always")
def update(value, seed):
    rotated = (value << np.uint64(17)) | (value >> np.uint64(47))
    return rotated + seed


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    for _ in range(iters):
        write_slot(buffer, slot, update(read_slot(buffer, slot), seed))


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    value = buffer[slot]
    for _ in range(iters):
        value = update(value, seed)
    buffer[slot] = value


def aligned_buffer(count, line):
    alignment = max(128, line)
    raw = np.zeros(count*8 + alignment, dtype=np.uint8)
    offset = -raw.ctypes.data % alignment
    buf = raw[offset:offset+count*8].view(np.uint64)
    assert buf.ctypes.data % alignment == 0
    return buf


def switches():
    if resource is None:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_nvcsw, r.ru_nivcsw


def run_stride_benchmark(case, ctx, seeds):
    private = case == "private"
    stride = 1 if private else int(case.split("_")[1])
    slots = [t*stride for t in range(NUM_THREADS)]
    line = ctx["cache_line_bytes"]
    buf = aligned_buffer(slots[-1]+1, line)
    worker = worker_private_accumulate if private else worker_strided_accumulate
    errors = []
    stamp = {}

    def start_clock():
        stamp["switches"] = switches()
        stamp["start"] = time.perf_counter_ns()

    def stop_clock():
        stamp["stop"] = time.perf_counter_ns()
        stamp["end_switches"] = switches()

    start = threading.Barrier(NUM_THREADS, action=start_clock)
    finish = threading.Barrier(NUM_THREADS, action=stop_clock)

    def run(t):
        try:
            if ctx["pinning"]:
                cpu = ctx["selected_cpus"][t]
                os.sched_setaffinity(0, {cpu})
                if os.sched_getaffinity(0) != {cpu}:
                    raise RuntimeError("Worker affinity verification failed")
            start.wait()
            worker(buf, slots[t], ITERATIONS_PER_THREAD, seeds[t])
            finish.wait()
        except BaseException as err:
            errors.append(err)
            start.abort()
            finish.abort()

    threads = [threading.Thread(target=run, args=(t,)) for t in range(NUM_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("Worker failed") from errors[0]
    ctx_delta = None
    if stamp["switches"] is not None:
        ctx_delta = [b-a for a,b in zip(stamp["switches"], stamp["end_switches"])]
    return dict(wall_ns=stamp["stop"]-stamp["start"], switches_voluntary_involuntary=ctx_delta,
                lines_relative_to_base=[s*8//line for s in slots],
                final_counters=[int(buf[s]) for s in slots])


def native_loop_ir(kernel):
    sig = kernel.signatures[0]
    module = binding.parse_assembly(kernel.inspect_llvm(sig))
    fn = module.get_function(kernel.overloads[sig].fndesc.llvm_func_name)
    blocks = {b.name:list(b.instructions) for b in fn.blocks}
    edges = {b:[v.name for v in ins[-1].operands if v.value_kind == binding.ValueKind.basic_block]
             for b,ins in blocks.items()}
    def in_cycle(start):
        todo, seen = list(edges[start]), set()
        while todo:
            nxt = todo.pop()
            if nxt == start:
                return True
            if nxt not in seen:
                seen.add(nxt)
                todo.extend(edges[nxt])
        return False
    loop = [str(i).strip() for b,ins in blocks.items() if in_cycle(b) for i in ins]
    return str(fn), loop


def audit_llvm_and_parity(results):
    reference = results["private"][0]["final_counters"]
    if any(r["final_counters"] != reference for runs in results.values() for r in runs):
        raise RuntimeError("Bit-exact parity failed")
    native, loop = native_loop_ir(worker_strided_accumulate)
    _, private_loop = native_loop_ir(worker_private_accumulate)
    loads = [s for s in loop if "load volatile i64" in s]
    stores = [s for s in loop if "store volatile i64" in s]
    private_stores = [s for s in private_loop if re.search(r"\bstore\b",s)]
    if not (loads and stores and not private_stores and worker_strided_accumulate.nopython_signatures
            and not re.search(r"@(?:Py[A-Z_]|_Py)", native)):
        raise RuntimeError("LLVM native loop audit failed")
    return dict(nopython=True, volatile_i64_loop_load_sites=len(loads),
                volatile_i64_loop_store_sites=len(stores), private_loop_store_sites=len(private_stores),
                load_example=loads[0], store_example=stores[0], parity=True)


def write_analysis(m):
    med = m["median_wall_ns"]
    total = NUM_THREADS*ITERATIONS_PER_THREAD
    table = "\n".join(f"| {case} | {med[case]:,} | {med[case]/total:.5f} | {med[case]/med['stride_8']:.4f} |"
                      for case in CASES)
    d = med["stride_16"]-med["stride_8"]
    a = m["llvm_parity_audit"]
    text = f"""# Cache lines and false sharing

Student 230103054 · GitHub `{GIT_USERNAME}`

Host: {m['cpu_model']}. The OS reports {m['logical_cores']} logical CPUs. The selected logical CPU IDs were {m['selected_cpus']}. Worker pinning enabled: {m['pinning']}. The line size used was {m['cache_line_bytes']} bytes ({m['cache_line_source']}). Virtualized topology describes the guest view.

## Q1. Who owns the line?

With a 64-byte line, byte offsets 0 and 8 map to the same line. Starting cold, if core 0 first reads its counter and no other cache holds it, the line can enter Exclusive on core 0. Its store then changes E to M silently. A store miss requesting ownership directly can instead take I to M. Suppose core 1 now stores to its own counter while core 0 holds M. Core 1 requests ownership (RFO); core 0 supplies or writes back the modified data and invalidates its copy. After invalidation acknowledgements, core 1 owns the line in M and its buffered store can drain. The store instruction may retire before ownership arrives; sustained delay can fill the store buffer. Reversing the writers reverses ownership.

If core 1 first reads the line for its read-modify-write while core 0 holds M, both can become S under MESI. Core 1 then requests an upgrade, invalidates core 0 and enters M. Stores already hitting M are local. The stride-1 median was {med['stride_1']:,} ns. Timing alone does not identify an individual coherence message.

## Q2. How much did spacing change throughput?

Each measurement covers {NUM_THREADS} workers × {ITERATIONS_PER_THREAD:,} updates = {total:,} updates. The table uses the median of {REPEATS} trials per layout. The clock starts at the release of the ready workers and stops when all finish. Compilation, allocation and thread creation are outside that interval.

| Layout | Median wall time (ns) | Wall ns per update | Time / stride 8 |
|---|---:|---:|---:|
{table}

The stride-1 / stride-8 slowdown factor is {med['stride_1']/med['stride_8']:.4f}×. These ns/update values describe aggregate throughput across four concurrent workers. For the private baseline the denominator counts arithmetic updates; only four final stores reach the shared array. Stride 8 / private is {med['stride_8']/med['private']:.4f}×. That comparison includes the repeated load, dependent store and store-to-load forwarding on a local line. Stride 1 / stride 8 additionally reflects competition for ownership. Neither ratio independently measures a hardware latency.

## Q3. Does a second line of padding help?

Stride 8 took {med['stride_8']:,} ns and stride 16 took {med['stride_16']:,} ns. Doubling the spacing made the measured median {abs(d):,} ns {'longer' if d>0 else 'shorter' if d<0 else 'different by zero'}, a signed change of {100*d/med['stride_8']:+.2f}%. With 64-byte lines each writer already has its own line at stride 8. A processor with an active adjacent-line prefetcher may fetch the other 64-byte half of an aligned 128-byte pair. If that read installs a shared copy on a peer, the next write to the prefetched line can require another invalidation. With a 128-byte-aligned base, stride 16 puts the four targets in separate pairs. Prefetch policy depends on the CPU and workload. No prefetch counters were sampled, so the timing difference cannot establish that this mechanism was active.

## Q4. What keeps the store in the loop?

The recurrence rotates the preceding 64-bit value by 17 bits and adds the worker seed modulo 2⁶⁴. Since the slot address is constant, ordinary loads and stores could be replaced with a register accumulator and one final write. The value dependency preserves the sequence of calculations but does not force memory traffic. The strided kernel therefore emits volatile native loads and stores. Each iteration must read and write the slot. Volatile imposes this compiler constraint; it does not synchronize threads. Each worker owns a distinct word.

The compiled native loop contains {a['volatile_i64_loop_load_sites']} volatile i64 load sites and {a['volatile_i64_loop_store_sites']} volatile i64 store sites across cyclic blocks, including any unrolled copies. Two instructions from this run:

```llvm
{a['load_example']}
{a['store_example']}
```

The audit inspected the nopython function separately from its Python-call wrapper. The private loop had {a['private_loop_store_sites']} store sites. All four final counters matched bit for bit across every trial.

## Q5. Why does the GIL change the experiment?

On a conventional GIL-enabled CPython build, only one thread executes Python bytecode at a time. Four threads doing scalar Python updates take turns. Interpreter dispatch, boxed arithmetic and GIL handoffs can make a fixed amount of work slower as threads are added. Core migration can still move cache lines, but this is a different experiment from four simultaneous native writers. Here `@njit(nogil=True)` lets the four OS threads execute compiled loops concurrently. Stride 1 took {med['stride_1']:,} ns with the GIL released. There was no pure-Python timing trial.
"""
    (OUT/"ANALYSIS.md").write_text(text, encoding="utf-8")


def main():
    ctx = init_system_context(GIT_USERNAME)
    seeds = [np.uint64((ctx["instance_seed"] + 2*t) & MASK | 1) for t in range(NUM_THREADS)]
    # Warm-up checks arithmetic against an independent small Python calculation.
    for worker in (worker_strided_accumulate, worker_private_accumulate):
        buf = aligned_buffer(4, ctx["cache_line_bytes"])
        for t, seed in enumerate(seeds):
            worker(buf, t, 103, seed)
            value = 0
            for _ in range(103):
                value = (((value << 17) | (value >> 47)) + int(seed)) & MASK
            assert int(buf[t]) == value
    results = {case:[] for case in CASES}
    for rep in range(REPEATS):
        for case in CASES[rep:]+CASES[:rep]:
            trial = run_stride_benchmark(case, ctx, seeds)
            results[case].append(trial)
            print(f"Trial {rep+1} {case}: {trial['wall_ns']:,} ns", flush=True)
    audit = audit_llvm_and_parity(results)
    manifest = dict(**ctx, num_threads=NUM_THREADS, iterations_per_thread=ITERATIONS_PER_THREAD,
                    repeats=REPEATS, thread_seeds=[int(s) for s in seeds], trials=results,
                    median_wall_ns={s:statistics.median(t["wall_ns"] for t in ts) for s,ts in results.items()},
                    llvm_parity_audit=audit, verification_passed=True)
    write_analysis(manifest)
    (OUT/"cache_lab_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("verification_passed: true")


if __name__ == "__main__":
    main()
