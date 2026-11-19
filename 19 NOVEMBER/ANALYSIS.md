# Cache lines and false sharing

Student 230103054 · GitHub `ibragimovmardandev-tech`

Host: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel. The OS reports 20 logical CPUs. The selected logical CPU IDs were [0, 1, 2, 3]. Worker pinning enabled: False. The line size used was 64 bytes (specification fallback). Virtualized topology describes the guest view.

## Q1. Who owns the line?

With a 64-byte line, byte offsets 0 and 8 map to the same line. Starting cold, if core 0 first reads its counter and no other cache holds it, the line can enter Exclusive on core 0. Its store then changes E to M silently. A store miss requesting ownership directly can instead take I to M. Suppose core 1 now stores to its own counter while core 0 holds M. Core 1 requests ownership (RFO); core 0 supplies or writes back the modified data and invalidates its copy. After invalidation acknowledgements, core 1 owns the line in M and its buffered store can drain. The store instruction may retire before ownership arrives; sustained delay can fill the store buffer. Reversing the writers reverses ownership.

If core 1 first reads the line for its read-modify-write while core 0 holds M, both can become S under MESI. Core 1 then requests an upgrade, invalidates core 0 and enters M. Stores already hitting M are local. The stride-1 median was 156,580,900 ns. Timing alone does not identify an individual coherence message.

## Q2. How much did spacing change throughput?

Each measurement covers 4 workers × 60,000,000 updates = 240,000,000 updates. The table uses the median of 3 trials per layout. The clock starts at the release of the ready workers and stops when all finish. Compilation, allocation and thread creation are outside that interval.

| Layout | Median wall time (ns) | Wall ns per update | Time / stride 8 |
|---|---:|---:|---:|
| stride_1 | 156,580,900 | 0.65242 | 1.0761 |
| stride_4 | 154,625,700 | 0.64427 | 1.0627 |
| stride_8 | 145,504,700 | 0.60627 | 1.0000 |
| stride_16 | 143,915,700 | 0.59965 | 0.9891 |
| private | 39,722,200 | 0.16551 | 0.2730 |

The stride-1 / stride-8 slowdown factor is 1.0761×. These ns/update values describe aggregate throughput across four concurrent workers. For the private baseline the denominator counts arithmetic updates; only four final stores reach the shared array. Stride 8 / private is 3.6631×. That comparison includes the repeated load, dependent store and store-to-load forwarding on a local line. Stride 1 / stride 8 additionally reflects competition for ownership. Neither ratio independently measures a hardware latency.

## Q3. Does a second line of padding help?

Stride 8 took 145,504,700 ns and stride 16 took 143,915,700 ns. Doubling the spacing made the measured median 1,589,000 ns shorter, a signed change of -1.09%. With 64-byte lines each writer already has its own line at stride 8. A processor with an active adjacent-line prefetcher may fetch the other 64-byte half of an aligned 128-byte pair. If that read installs a shared copy on a peer, the next write to the prefetched line can require another invalidation. With a 128-byte-aligned base, stride 16 puts the four targets in separate pairs. Prefetch policy depends on the CPU and workload. No prefetch counters were sampled, so the timing difference cannot establish that this mechanism was active.

## Q4. What keeps the store in the loop?

The recurrence rotates the preceding 64-bit value by 17 bits and adds the worker seed modulo 2⁶⁴. Since the slot address is constant, ordinary loads and stores could be replaced with a register accumulator and one final write. The value dependency preserves the sequence of calculations but does not force memory traffic. The strided kernel therefore emits volatile native loads and stores. Each iteration must read and write the slot. Volatile imposes this compiler constraint; it does not synchronize threads. Each worker owns a distinct word.

The compiled native loop contains 9 volatile i64 load sites and 9 volatile i64 store sites across cyclic blocks, including any unrolled copies. Two instructions from this run:

```llvm
%.154.prol = load volatile i64, ptr %sunkaddr1, align 8
store volatile i64 %.159.prol, ptr %sunkaddr1, align 8
```

The audit inspected the nopython function separately from its Python-call wrapper. The private loop had 0 store sites. All four final counters matched bit for bit across every trial.

## Q5. Why does the GIL change the experiment?

On a conventional GIL-enabled CPython build, only one thread executes Python bytecode at a time. Four threads doing scalar Python updates take turns. Interpreter dispatch, boxed arithmetic and GIL handoffs can make a fixed amount of work slower as threads are added. Core migration can still move cache lines, but this is a different experiment from four simultaneous native writers. Here `@njit(nogil=True)` lets the four OS threads execute compiled loops concurrently. Stride 1 took 156,580,900 ns with the GIL released. There was no pure-Python timing trial.
