"""Complete metadata for the observed i7-12700H / 2x8 GiB DDR5-4800 host.
Run from lab_submission. Preserves timings and arrays; reruns verification only.
"""
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCES = {
    'intel_cpu': 'https://www.intel.com/content/www/us/en/products/sku/132228/intel-core-i712700h-processor-24m-cache-up-to-4-70-ghz/specifications.html',
    'base_frequencies_oem': 'https://psref.lenovo.com/product/legion_5_pro_16iah7h?tab=spec',
    'asus_memory': 'https://rog.asus.com/laptops/rog-zephyrus/rog-zephyrus-m16-2022-series/spec/',
    'fma_instruction_throughput': 'https://uops.info/html-instr/VFMADD231PD_YMM_YMM_YMM.html',
}


def enrich(host, roof, rows):
    if 'i7-12700H' not in host.get('cpu_model', '') or host.get('physical_cores') != 14:
        raise ValueError('This metadata supplement is specifically for the observed i7-12700H with 14 cores.')
    host['microarchitecture'] = 'Intel Alder Lake (hybrid P-core/E-core design)'
    host['base_frequency_mhz'] = 2300.0
    host['base_frequency_scope'] = 'Nominal P-core base frequency from OEM specifications, not an OS measurement; E-core base is 1700 MHz.'
    host['base_frequency_mhz_by_core_type'] = {'P': 2300.0, 'E': 1700.0}
    host['core_counts_by_type'] = {'P': 6, 'E': 8}
    host['system_model'] = 'ASUSTeK COMPUTER INC. ROG Zephyrus M16 GU603ZEB_GU603ZE'
    host['memory_configuration'] = {
        'source': 'User-provided Win32_PhysicalMemory output, 2026-10-29; two populated 8 GiB memory devices, each reporting 4800 and DataWidth=64.',
        'technology': 'DDR5', 'smbios_memory_type': 34,
        'device_capacity_bytes': [8589934592, 8589934592],
        'configured_data_rate_mt_s': 4800,
        'device_data_width_bits': [64, 64],
        'part_number': 'MTC4C10163S1SC48BA1',
        'modeled_aggregate_data_width_bits': 128,
        'channel_basis': 'Dual 64-bit-equivalent data paths inferred from the two populated controllers and ASUS dual-channel support; channel interleaving was not independently measured. DDR5 divides each 64-bit path into two 32-bit subchannels.',
        'layout_note': 'ASUS specifies onboard memory plus one SO-DIMM; two SMBIOS devices do not imply two removable modules.'
    }
    old = host.get('warnings', [])
    host.setdefault('original_profiling_warnings', list(old))
    resolved_prefixes = ('Base frequency is not exposed;', 'Microarchitecture is not reliably exposed')
    host['warnings'] = [w for w in old if not w.startswith(resolved_prefixes)
                        and not (w.startswith('Windows topology uses current processor group;') and host['logical_threads'] <= 64)]
    host['metadata_supplement'] = {'sources': SOURCES,
        'note': 'Original measured frequency_snapshot_mhz and recorded_utc are unchanged. Supplemental model/specification data is distinguished from observations.',
        'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    bw = 4800 * 128 / 8 / 1000
    fp64 = 6 * 2.3 * 16 + 8 * 1.7 * 8
    roof.update({
        'peak_theoretical_bandwidth_gbs': bw,
        'peak_theoretical_fp64_gflops': fp64,
        'ridge_point_flops_per_byte': fp64 / bw,
        'classification': 'Memory-side limited in the nominal DRAM roofline model; actual runtime bottleneck is not established, especially for the cache-resident strong case',
        'hardware_model': {
            'bandwidth_equation': '4800e6 transfers/s * 128 bits / 8 = 76.8e9 bytes/s',
            'bandwidth_assumption': host['memory_configuration']['channel_basis'],
            'compute_equation': '6 P-cores * 2.3 GHz * 16 FP64 FLOP/cycle + 8 E-cores * 1.7 GHz * 8 FP64 FLOP/cycle = 329.6 GFLOP/s',
            'compute_basis': '256-bit packed FP64 FMA performs 4 lanes * 2 operations = 8 FLOP/instruction; reported reciprocal throughput is 0.5 cycle on Alder Lake-P and 1 cycle on Alder Lake-E.',
            'compute_scope': 'Nominal-base-frequency FMA model, not a measured sustained peak, not a turbo maximum, and not the instruction mix of the five-point kernel.',
            'sources': SOURCES,
            'measured_dram_bandwidth_gbs': None,
            'measured_sustained_compute_peak_gflops': None,
        },
        'modeled_dram_roofs_gflops': {
            'no_reuse': bw * roof['ai_flops_per_byte_no_reuse'],
            'write_allocate': bw * roof['ai_flops_per_byte_write_allocate'],
            'ideal_reuse': bw * roof['ai_flops_per_byte_ideal_reuse'],
        },
    })
    strong = [r for r in rows if r['mode'] == 'strong']
    best = max(strong, key=lambda r: float(r['speedup']))
    l3 = max((c['size_bytes'] for c in host['caches'] if c['level'] == 3), default=0)
    roof['cache_interpretation'] = {
        'strong_two_buffer_bytes': roof['strong_base_working_set_bytes'],
        'l3_bytes': l3,
        'note': 'The two strong-scaling buffers occupy 20.25 MiB versus 24 MiB L3. Potential cache residence means a DRAM-only roofline is not sufficient. Other working data and cache contention can reduce usable capacity.'
    }
    roof['measured_scaling_interpretation'] = {
        'best_strong_processes': int(best['P']), 'best_strong_speedup': float(best['speedup']),
        'amdahl_fit_caveat': 'The global fit has normalized runtime RMSE about 0.17. Its 3.49x asymptote is below the observed 3.94x speedup; do not interpret it as a physical ceiling. A fixed serial fraction does not explain the nonmonotonic scaling.',
        'possible_overheads': 'Barriers, cache/memory contention, scheduling, power limits and heterogeneous cores can affect scaling; their separate contributions were not measured.'
    }
    note = 'Roofline ceilings are now explicit model estimates at nominal base clocks and inferred dual-channel memory configuration, not measured hardware limits. The FMA compute roof is a generic capacity model.'
    roof['limitations'] = [x for x in roof.get('limitations', []) if not x.startswith('Peak bandwidth needs memory data rate')]
    if note not in roof['limitations']:
        roof['limitations'].append(note)
    return host, roof


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def main():
    root = Path.cwd()
    if not (root / 'run_lab.py').is_file():
        raise SystemExit('Open the terminal in lab_submission (where run_lab.py is located), then run this script.')
    results = root / 'results'
    host_path = results / 'host_profile.json'
    roof_path = results / 'roofline_summary.json'
    host = json.loads(host_path.read_text(encoding='utf-8-sig'))
    roof = json.loads(roof_path.read_text(encoding='utf-8-sig'))
    with (results / 'scaling_metrics.tsv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    params = json.loads((results / 'salted_params.json').read_text(encoding='utf-8-sig'))
    host, roof = enrich(host, roof, rows)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = root / 'metadata_backups' / stamp
    backup.mkdir(parents=True)
    for path in (host_path, roof_path, results / 'verification_receipt.sig', root / 'README.md'):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    write_json(host_path, host)
    write_json(roof_path, roof)
    print('Metadata updated. Benchmark arrays and timings were not changed.', flush=True)
    print('Modeled bandwidth: 76.8 GB/s; nominal FP64 FMA capacity: 329.6 GFLOP/s.', flush=True)
    print('Re-running M4 verification only. Keep other large programs closed.', flush=True)
    subprocess.run([sys.executable, '-u', str(root / 'run_lab.py'), '--student-id', params['student_id'], '--stage', '4'], cwd=root, check=True)
    readme = root / 'README.md'
    caveat = (' The global Amdahl fit is approximate: its fitted 3.49x asymptote is below the observed 3.94x speedup at four processes. '
              'The two strong-scaling buffers occupy 20.25 MiB versus 24 MiB L3, so actual DRAM saturation is not established. '
              'Roofline uses a nominal-base-frequency FMA capacity model and inferred dual-channel DDR5-4800 bandwidth, not measured sustained ceilings.')
    text = readme.read_text(encoding='utf-8').rstrip()
    if caveat.strip() not in text:
        readme.write_text(text + caveat + '\n', encoding='utf-8')
    print('DONE. Updated host_profile.json, roofline_summary.json, README.md and verification_receipt.sig.', flush=True)
    print('Original metadata backup:', backup, flush=True)
    print('No Git commands were executed. Do not rerun M0/M3 unless you intend to regenerate the original metadata.', flush=True)


if __name__ == '__main__':
    main()
