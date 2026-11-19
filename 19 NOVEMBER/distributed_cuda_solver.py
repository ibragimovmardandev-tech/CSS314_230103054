import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = GLOBAL_NY = 512
N = GLOBAL_NX // 2
DX = 1 / (GLOBAL_NX + 1)
STEPS = 400
LAMBDA_DAMP = 0.15
OPPOSITE = (1, 0, 3, 2)


def get_gpu_uuid():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode()).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(user):
    world = MPI.COMM_WORLD
    if world.size != 4:
        raise ValueError("Launch exactly four MPI ranks.")
    cuda.select_device(0)
    cart = world.Create_cart([2, 2], periods=[False, False], reorder=False)
    uuid = cart.bcast(get_gpu_uuid() if cart.rank == 0 else None, root=0)
    return cart, (*cart.Shift(0, 1), *cart.Shift(1, 1)), uuid, derive_student_seed(user, uuid)


def initialize_subdomains(coords, neighbors, seed):
    row, col = coords
    y = (row * N + np.arange(N + 2)[:, None]) * DX
    x = (col * N + np.arange(N + 2)[None, :]) * DX
    rng = np.random.default_rng(seed)
    k = 0.16 + 0.08 * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
    u = np.zeros_like(k)
    for _ in range(3):
        cx, cy = rng.uniform(0.15, 0.85, 2)
        u += rng.uniform(0.8, 1.4) * np.exp(-((x-cx)**2 + (y-cy)**2)/0.035)
    s = 0.0001 * np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.01)
    for side, neighbor in enumerate(neighbors):
        if neighbor == MPI.PROC_NULL:
            if side < 2:
                u[(0, -1)[side], :] = 0
            else:
                u[:, (0, -1)[side-2]] = 0
    return u, k, s


@cuda.jit
def pack_halos_kernel(u, buf):
    t = cuda.grid(1)
    if t < N:
        buf[t] = u[1, t+1]
        buf[N+t] = u[N, t+1]
        buf[2*N+t] = u[t+1, 1]
        buf[3*N+t] = u[t+1, N]


@cuda.jit
def unpack_halos_kernel(u, buf):
    t = cuda.grid(1)
    if t < N:
        u[0, t+1] = buf[t]
        u[N+1, t+1] = buf[N+t]
        u[t+1, 0] = buf[2*N+t]
        u[t+1, N+1] = buf[3*N+t]


@cuda.jit
def tiled_stencil_kernel(u, out, k, source, dt, r_start, r_end, c_start, c_end):
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sx, sy = tx+1, ty+1
    j = c_start + cuda.blockIdx.x*TILE_DIM + tx
    i = r_start + cuda.blockIdx.y*TILE_DIM + ty
    # Padding lanes load the positive apron for a partial block. They join
    # the barrier but never read an MPI ghost while the deep interior runs.
    r, c = min(i, r_end+1), min(j, c_end+1)
    tile[sy, sx] = u[r, c]
    if tx == 0:
        tile[sy, 0] = u[r, c-1]
    if ty == 0:
        tile[0, sx] = u[r-1, c]
    if tx == 15:
        tile[sy, sx+1] = u[r, min(c+1, c_end+1)]
    if ty == 15:
        tile[sy+1, sx] = u[min(r+1, r_end+1), c]
    cuda.syncthreads()
    if i <= r_end and j <= c_end:
        v = tile[sy, sx]
        kc = k[i, j]
        flux = ((kc+k[i, j+1])*(tile[sy, sx+1]-v)
                + (kc+k[i, j-1])*(tile[sy, sx-1]-v)
                + (kc+k[i+1, j])*(tile[sy+1, sx]-v)
                + (kc+k[i-1, j])*(tile[sy-1, sx]-v))
        out[i, j] = v + dt*(0.5*flux/(DX*DX) - LAMBDA_DAMP*v*v*v + source[i, j])


def exchange_halos_async(cart, neighbors, u, boundary, send, host_send, host_recv):
    pack_halos_kernel[2, 128, boundary](u, send)
    send.copy_to_host(host_send, stream=boundary)
    boundary.synchronize()
    requests = []
    for side, neighbor in enumerate(neighbors):
        section = slice(side*N, (side+1)*N)
        requests.append(cart.Irecv(host_recv[section], neighbor, OPPOSITE[side]))
        requests.append(cart.Isend(host_send[section], neighbor, side))
    return requests


def launch_box(args, stream, r0, r1, c0, c1):
    grid = ((c1-c0+16)//16, (r1-r0+16)//16)
    tiled_stencil_kernel[grid, (16, 16), stream](*args, r0, r1, c0, c1)


def step_overlapped(cart, neighbors, args, streams, buffers, overlap):
    interior, boundary = streams
    send, recv, host_send, host_recv = buffers
    requests = exchange_halos_async(cart, neighbors, args[0], boundary, send, host_send, host_recv)
    if overlap:
        launch_box(args, interior, 2, N-1, 2, N-1)
    MPI.Request.Waitall(requests)
    recv.copy_to_device(host_recv, stream=boundary)
    unpack_halos_kernel[2, 128, boundary](args[0], recv)
    if not overlap:
        launch_box(args, boundary, 2, N-1, 2, N-1)
    for box in ((1, 1, 1, N), (N, N, 1, N), (2, N-1, 1, 1), (2, N-1, N, N)):
        launch_box(args, boundary, *box)
    interior.synchronize()
    boundary.synchronize()


def metrics(cart, current, previous, step):
    a = current.copy_to_host()[1:-1, 1:-1]
    b = previous.copy_to_host()[1:-1, 1:-1]
    local = np.array([a.sum()*DX**2, ((a-b)**2).sum()*DX**2])
    total = np.empty_like(local)
    cart.Allreduce(local, total, op=MPI.SUM)
    return dict(step=step, energy=float(total[0]), l2_residual=float(np.sqrt(total[1])))


def run_solver(user):
    cart, neighbors, uuid, seed = init_distributed_context(user)
    coords = cart.Get_coords(cart.rank)
    u0, kh, sh = initialize_subdomains(coords, neighbors, seed)
    kmax = cart.allreduce(kh[1:-1, 1:-1].max(), op=MPI.MAX)
    dt = 0.8 * DX**2 / (4*kmax)
    k, source = cuda.to_device(kh), cuda.to_device(sh)
    streams = cuda.stream(), cuda.stream()
    buffers = (cuda.device_array(4*N, np.float64), cuda.device_array(4*N, np.float64),
               cuda.pinned_array(4*N, np.float64), cuda.pinned_array(4*N, np.float64))
    buffers[-1].fill(0)  # PROC_NULL receives leave the destination unchanged.
    print(f"rank={cart.rank} coords={coords} neighbors={neighbors} shape={u0.shape}", flush=True)
    record = dict(github_username=user, student_id="230103054", gpu_uuid=uuid,
                  instance_seed=seed, mpi_ranks=cart.size, cart_dims=[2, 2],
                  local_shape=list(u0.shape), halo_bytes=N*u0.itemsize, steps=STEPS, dx=DX, dt=dt, kappa_max=kmax)
    # Warm all kernels and both execution paths before measuring.
    a, b = cuda.to_device(u0), cuda.to_device(u0)
    for mode in (False, True):
        step_overlapped(cart, neighbors, (a, b, k, source, dt), streams, buffers, mode)
    ptx = next(iter(tiled_stencil_kernel.inspect_asm().values()))
    counts = {op: len(re.findall(r"\b"+re.escape(op)+r"\b", ptx))
              for op in ("st.shared.f64", "ld.shared.f64", "bar.sync", "st.shared", "ld.shared")}
    record.update(ptx_counts=counts, ptx_audit_passed=all(counts[t] > 0 for t in ("st.shared", "ld.shared", "bar.sync")))
    for mode, label in ((False, "serialized"), (True, "overlapped")):
        a.copy_to_device(u0)
        b.copy_to_device(u0)
        trace = [metrics(cart, a, b, 0)]
        wall_ms = gpu_ms = 0.
        start_event, stop_event = cuda.event(), cuda.event()
        for start in range(0, STEPS, 100):
            cart.Barrier()
            start_event.record(streams[1])
            tick = perf_counter()
            for _ in range(100):
                step_overlapped(cart, neighbors, (a, b, k, source, dt), streams, buffers, mode)
                a, b = b, a
            wall_ms += (perf_counter()-tick)*1000
            stop_event.record(streams[1])
            stop_event.synchronize()
            gpu_ms += cuda.event_elapsed_time(start_event, stop_event)
            trace.append(metrics(cart, a, b, start+100))
        record[label+"_ms"] = cart.allreduce(wall_ms, op=MPI.MAX)
        record[label+"_cuda_event_ms"] = cart.allreduce(gpu_ms, op=MPI.MAX)
        record[label+"_trace"] = trace
        final = a.copy_to_host()[1:-1, 1:-1]
        if not mode:
            baseline = final.copy()
    diff = cart.allreduce(float(np.max(np.abs(final-baseline))), op=MPI.MAX)
    record["parity_max_diff"] = diff
    if not record["ptx_audit_passed"] or not diff < 1e-12:
        raise RuntimeError(f"Verification failed: {counts}, parity={diff}")
    if not all(np.isfinite(item["energy"]) and np.isfinite(item["l2_residual"])
               for item in trace):
        raise RuntimeError("Non-finite diagnostics")
    if cart.rank == 0:
        output = json.dumps(record, indent=2, allow_nan=False)
        print(output, flush=True)
        Path("submission_manifest.json").write_text(output+"\n")


if __name__ == "__main__":
    run_solver(sys.argv[1])
