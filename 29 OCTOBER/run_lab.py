import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
TAGS=['M0-ENV','M1-LOCALITY','M2-SHARING','M3-SCALING','M4-FINAL']
MESSAGES=['M0: Environment and salting configuration','M1: Memory stride and loop interchange benchmarking',
          'M2: False sharing reproduction and Karp-Flatt analysis','M3: Strong and weak scaling evaluation',
          'M4: Final verification and completed practicum submission']


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()


def main():
    parser=argparse.ArgumentParser(description='Run actual local experiments. No synthetic measurements.')
    parser.add_argument('--student-id',required=True)
    parser.add_argument('--stage',type=int,choices=range(5),help='Only this milestone; omit for 0..4')
    parser.add_argument('--git',action='store_true',help='Create a real commit and tag after each milestone. Never pushes.')
    parser.add_argument('--smoke',action='store_true',help='Small functional test in results_smoke; not a submission.')
    parser.add_argument('--peak-bandwidth-gbs',type=float)
    parser.add_argument('--peak-gflops',type=float)
    args=parser.parse_args()
    if args.smoke and args.git:
        parser.error('Smoke tests cannot create submission commits.')
    for peak in (args.peak_bandwidth_gbs,args.peak_gflops):
        if peak is not None and peak<=0:
            parser.error('Peak ceilings must be positive.')
    if args.smoke:
        os.environ['LAB_SMOKE']='1'
    sys.path.insert(0,str(ROOT/'src'))
    from common import RESULTS, read_json, write_json
    from salt import get_student_params
    get_student_params(args.student_id)
    stages=[args.stage] if args.stage is not None else list(range(5))
    if args.git:
        try:
            git('rev-parse','--show-toplevel')
            git('var','GIT_AUTHOR_IDENT')
            for stage in stages:
                if git('tag','--list',TAGS[stage]):
                    raise RuntimeError(f'Tag {TAGS[stage]} already exists. Do not overwrite tags; resume with --stage for an uncompleted milestone.')
            if git('diff','--cached','--name-only'):
                raise RuntimeError('Git index contains staged changes. Commit or unstage them yourself before --git.')
        except (subprocess.CalledProcessError,FileNotFoundError) as exc:
            raise RuntimeError('Open this project inside your cloned Git repository, and configure git user.name/user.email.') from exc
    if stages[0]>0:
        if read_json('salted_params.json')['student_id']!=args.student_id:
            raise RuntimeError('Student ID differs from existing results.')
    for stage in stages:
        print(f'\n=== {TAGS[stage]} {"SMOKE ONLY" if args.smoke else "LOCAL MEASUREMENTS"} ===',flush=True)
        if stage==0:
            from profile_env import profile
            profile(args.student_id)
            if args.smoke:
                p=read_json('salted_params.json'); p.update(N=64,buffer_elements=65_536)
                write_json('salted_params.json',p)
        elif stage==1:
            import stride_bench,dense_kernel
            stride_bench.run(); dense_kernel.run()
        elif stage==2:
            import false_sharing
            false_sharing.run(k=20_000 if args.smoke else 5_000_000,repeats=1 if args.smoke else 3)
        elif stage==3:
            import scaling_study
            scaling_study.run(steps=3 if args.smoke else 30,repeats=1 if args.smoke else 3,
                max_cores=2 if args.smoke else None,peak_bandwidth=args.peak_bandwidth_gbs,peak_gflops=args.peak_gflops)
        elif stage==4:
            from verify_outputs import verify
            receipt=verify(args.smoke)
            if not args.smoke:
                dense=read_json('loop_interchange.json'); roof=read_json('roofline_summary.json')
                import csv
                with (RESULTS/'false_sharing_karp_flatt.tsv').open() as f:
                    fs=list(csv.DictReader(f,delimiter='\t'))
                ratio=next(float(r['padded_recovery_ratio']) for r in fs if r['mode']=='padded' and int(r['P'])==8)
                paragraph=(f'# CS482 local benchmark findings\n\nStudent {args.student_id}, CPU {read_json("host_profile.json")["cpu_model"]}. '
                    f'The measured i-j-k/i-k-j runtime ratio was {dense["speedup"]:.3f}x with {dense["backend"]}; '
                    f'at P=8, unpadded/padded time was {ratio:.3f}x. '
                    f'The constrained Amdahl serial-fraction fit was {roof["amdahl_serial_fraction_constrained"]}; '
                    f'roofline classification: {roof["classification"]}. '
                    'The five-point stencil has modeled arithmetic intensity between 5/48 and 5/16 FLOP/byte. '
                    'These measurements include synchronization and runtime overhead, so they do not isolate DRAM or MESI traffic. '
                    'The PDF Karp-Flatt values are reported literally for growing counter work; both the PDF weak-scaling quantity and conventional throughput speedup are retained. '
                    'All saved outputs passed independent numerical checks. See the receipt for unresolved telemetry or pathology warnings. '
                    'Implementation prepared with AI assistance; benchmarks are generated on the executing host.\n')
                (ROOT/'README.md').write_text(paragraph,encoding='utf-8')
        if args.git:
            git('add','--','src','results','README.md','START_HERE_RU.md','METHODS.md','requirements.txt','run_lab.py','verify_outputs.py','.gitignore')
            git('commit','-m',MESSAGES[stage]); git('tag',TAGS[stage])
            print(f'Created {TAGS[stage]} at {git("rev-parse","--short","HEAD")}',flush=True)
    if not args.git and not args.smoke:
        print('Measurements complete. Tags were not created; use the stage-by-stage Git instructions.')
    if args.smoke:
        print('SMOKE ONLY: outputs are in results_smoke and must not be submitted.')


if __name__=='__main__':
    mp.freeze_support()
    try:
        main()
    except (Exception,KeyboardInterrupt) as error:
        print(f'\nSTOPPED: {error}\nCompleted files were preserved; fix the cause and rerun the failed stage.',file=sys.stderr)
        raise
