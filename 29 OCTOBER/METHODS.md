# Measurement and verification notes

## Coverage

| PDF requirement | Implementation / evidence |
|---|---|
| CPU, physical/logical cores, cache sizes, line size, OS/Python | src/profile_env.py -> results/host_profile.json; unavailable fields explicitly null |
| Exact SHA-256 student salting | src/salt.py; eight ASCII digits required |
| 2^24 doubles; strides 1..128; 5 warmups + 5 samples | stride_results.tsv; medians and raw nanosecond samples |
| N-pitch matrices, M=N/2, i-j-k vs i-k-j, three trials | loop_interchange.json and dense_ijk.npy / dense_ikj.npy |
| P=1,2,4,8, K=5,000,000; three counter variants | false_sharing_karp_flatt.tsv and false_sharing_raw.json; three trials |
| Strong and weak stencil scaling to physical core count | scaling_metrics.tsv, scaling_raw.json and saved stencil arrays |
| Amdahl fit, ceiling, Gustafson comparison, AI | roofline_summary.json; measured versus modeled quantities distinguished |
| Salted SHA-256 and tolerance 1e-12 | independent verify_outputs.py; verification_receipt.sig |
| Five milestone commits and tags | run_lab.py --git, or documented manual commands |
| Brief findings paragraph | README.md generated at M4 from actual results |

## Scope of timing

Stride uses perf_counter_ns; durations have nanosecond units, not a claim of nanosecond hardware accuracy. The dense kernel excludes input allocation; the fallback includes output-list allocation/conversion and must not be directly compared with a different backend. All trials must use the same backend. Native compilation uses -O2, no fast-math and no FP contraction; verify both loop orders against independent NumPy matrix multiplication.

False sharing aligns the base counter to 64 bytes and uses a raw shared array (lock=False). Each process owns its own counter; padding changes stride from 1 to 8 int64 slots. The default multiprocessing.Array lock would serialize accesses globally and confound coherence with lock contention. Privatization takes a shared lock only for the final scalar store. The maximum worker-loop duration is the primary timer; total elapsed time including spawn and join is retained separately. Record actual affinity where supported. On Linux, one allowed logical CPU per distinct physical core is selected; on Windows masks are read from the OS. P=8 is still tested on smaller machines as required, with oversubscription disclosed by the profile. On machines with cache lines larger than 64 bytes, the assignment's 64-byte padding may not fully separate counters.

Stencil uses persistent processes within each trial, two shared buffers, zero boundary conditions and synchronized Jacobi steps. Each iteration uses 4 additions and 1 multiply per interior cell, without an extra global reduction. Rows form disjoint output regions. A barrier after every timestep protects the input halos before buffers swap. All P values use identical update order and seed. A single-process reference is recomputed independently of multiprocessing for each shape. Steps=30 and trials=3 for full runs. Primary time includes the barriers; end-to-end time also includes spawning. BLAS/OMP thread counts are set to one to avoid hidden nested parallelism.

## Two mathematical inconsistencies in the PDF

1. The false-sharing instructions fix **K per worker**, so total work is P*K. Classical Karp-Flatt assumes fixed work. The requested e(P)=(P*T(P)/T(1)-1)/(P-1) is reported as `karp_flatt_literal`, without calling it a valid serial fraction. With ideal constant weak runtime it is approximately 1, not 0. `throughput_speedup=P*T(1)/T(P)` and `weak_efficiency=T(1)/T(P)` are also reported. The pathology is assessed directly from unpadded/padded runtimes, with no manufactured collapse.
2. The PDF weak formula P*Work(1)*T(1)/(Work(P)*T(P)) approaches T(1)/T(P), an efficiency, when Work(P)≈P*Work(1). It is saved exactly as `scaled_speedup_pdf`. Conventional throughput scaling is `(Work(P)/Work(1))*T(1)/T(P)`. Actual work counts interior stencil updates, incorporating ceil and boundary effects.

Amdahl is fitted by least squares on normalized runtime T(P)/T(1), with an unconstrained estimate and an estimate constrained to [0,1]. Large residuals or out-of-range unconstrained estimates indicate that simple Amdahl assumptions do not explain the measurements. A zero fitted serial fraction gives an unbounded model ceiling (JSON null plus explanation), not a claim of infinite observed speedup. Gustafson P-s(P-1) uses that fitted serial fraction only as a comparison, not as an independently measured weak fraction.

## Roofline and telemetry limits

Five-point arithmetic intensity is 5/48 FLOP/B under 5 independent reads plus one store, 5/16 with ideal DRAM reuse and streaming stores, or 5/24 with write allocation. These are modeled bounds. NumPy makes multiple passes, and caches may serve the work; actual DRAM bytes cannot be inferred from useful bytes alone. Two arrays occupy 16*N*N bytes for strong scaling. The recorded cache sizes and working set allow comparisons, but do not establish cache hit rates.

CPU peak memory bandwidth depends on active channels and memory data rate; theoretical FP64 peak depends on microarchitecture, vector hardware and clock. These are not reliably discoverable via portable Python. Explicit user-supplied verified ceilings enable the roofline ridge-point comparison. Without them, classification is undetermined; actual bottlenecks may include barriers or Python/NumPy overhead. No PMU cache-hit telemetry is claimed. The PDF introductory paragraph mentions cache hit rates, but provides no portable counter command or output schema; PMU collection would require a separate OS-specific run and permissions if the instructor requires it.

## Integrity and provenance

The starter supplied in this conversation contains placeholders and no instructor verify_outputs.py. The bundled verifier is newly implemented and is labeled accordingly. It checks exact salted dimensions, complete benchmark configurations, counters, independent dense/stencil values, salted canonical little-endian rounded SHA-256 digests, and result/source-file hashes. The .sig file is a JSON integrity receipt, not a digital signature and not hardware attestation. It cannot certify authorship, physical execution, an official grade or the 180-minute window. Git timestamps come from actual local commits. No tags or measurements are included in the delivered archive.

The course PDF prohibits AI code generation. This implementation was prepared with AI assistance, which is disclosed in README. Review permission with the instructor; do not present it as unaided work.

## Validation performed before delivery

The small end-to-end suite passed with both the ctypes backend and the pure-Python fallback, including all 1/2/4/8 counter configurations and strong/weak stencil runs at 1 and 2 processes. The full-size M1 test used a test-only ID producing N=1088, M=544, all eight full-buffer strides and three matrix trials per loop order; numerical comparison passed. A separate disposable Git repository verified M0/M1 commits and tags, and protection against reusing existing tags. Deliberately corrupted counter data was rejected by the verifier. These tests ran in the preparation Linux environment, not on the student's Windows laptop; no preparation measurements or test commits are included. Full K=5,000,000 counter runs and the full physical-core stencil sweep still need to run locally.
