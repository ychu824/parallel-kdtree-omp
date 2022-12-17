#!/usr/bin/env python3

import sys
import os
import re
import numpy as np
import time

# Usage: ./checker.py v1/v2 0/1
version = int(sys.argv[1])
if version == 0:
    print('generate reference performance')
# load_balance = '-lb' if sys.argv[2] == '1' else ''
# prog = 'nbody-release-' + version
prog = 'build/kdtree'
if version == 0:
    workers = [0]
else:
    workers = [1, 2, 4, 8, 16, 32]

# scene_name, particle_num, space_size, iteration
scenes = (
    ('corner-50000',     50000,  500.0, 5),
    ('corner-200000',   200000, 1000.0, 5),
    ('diagonal-50000',   50000,  500.0, 5),
    ('diagonal-200000', 200000, 1000.0, 5),
    ('random-50000',     50000,  500.0, 5),
    ('random-200000',   200000, 1000.0, 5),
)

perf_events = [
    "cache-references",
    "cache-misses",
    "cycles",
    "bus-cycles",
    "context-switches",
    "task-clock",
    "page-faults",
]

perf_prefix = "perf stat -e " + ",".join(perf_events) + " -a "

ref_perfs = (
    (0.197067, 0.010336),
    (0.951315, 0.147518),
    (0.195, 0.00892),
    (0.954365, 0.119437),
    (0.194814, 0.008987),
    (0.958455, 0.159672),
)
ref_perfs = np.array(ref_perfs)

# build time, traverse time
# perfs = [[None for _ in range(2)] * len(workers) for _ in range(len(scenes))]
perfs = np.zeros((len(scenes), len(workers), 2))
# print(perfs)
# print(K)

def read_and_sort_pts(filepath: str, k: int):
    f = open(filepath)
    lines = f.readlines()
    # print(len(lines), k)
    assert len(lines) % (k + 1) == 0

    query = None
    pts = []
    cnt = 0
    while cnt < len(lines):
        tmp = []
        for line in lines[cnt: cnt+k]:
            splited = line.split(',')
            tag = int(splited[-1])
            pt = [float(x) for x in splited[:-1]]
            if tag == 2:
                query = pt
            else:
                tmp.append(pt)

        cnt += (k + 1)
        tmp.sort()
        pts.append(query)
        pts += tmp

    f.close()
    return pts


def compare(actual, ref, k):
    actual = read_and_sort_pts(actual, k)
    ref = read_and_sort_pts(ref, k)
    threshold = 1.0 if 'repeat' in actual else 0.1

    for i, (l1, l2) in enumerate(zip(actual, ref)):
        # l1 = [float(x) for x in l1.split()]
        # l2 = [float(x) for x in l2.split()]
        assert len(l1) == len(l2),\
            f'ERROR -- invalid format at line {i}, should contain {len(l2)} floats'
        assert all(abs(x - y) < threshold for x, y in zip(l1, l2)),\
            f'ERROR -- incorrect result at line {i}'

    # threshold = 1.0 if 'repeat' in actual else 0.1
    # actual = open(actual).readlines()
    # ref = open(ref).readlines()
    # assert len(actual) == len(ref), \
    #     f'ERROR -- number of particles is {len(actual)}, should be {len(ref)}'
    # for i, (l1, l2) in enumerate(zip(actual, ref)):
    #     l1 = [float(x) for x in l1.split()]
    #     l2 = [float(x) for x in l2.split()]
    #     assert len(l2) == 5 and len(l1) == len(l2),\
    #         f'ERROR -- invalid format at line {i}, should contain {len(l2)} floats'
    #     assert all(abs(x - y) < threshold for x, y in zip(l1, l2)),\
    #         f'ERROR -- incorrect result at line {i}'


def compute_score(actual, ref):
    # actual <= 1.2 * ref: full score
    # actual >= 3.0 * ref: zero score
    # otherwise: linear in (actual / ref)
    return min(max((3.0 - actual / ref) / 1.8, 0.0), 1.0)


def cal_speedup(actual, ref):
    # ref.shape = (# scenes, )
    # actual.shape = (# scenes, # workers)
    speedup = ref / actual.T
    speedup = speedup.T
    return speedup


def print_table(title: str, speedup: np.ndarray):
    print(f'\n{title}')
    header = '|'.join(f' {x:<15} ' for x in ['Scene Name'] + workers)
    print(header)
    print('-' * len(header))
    for scene, perf in zip(scenes, speedup):
        perf_str = '|'.join(f' {x:<15.3f} ' for x in perf)
        print(f' {scene[0]:<15} |{perf_str}')


# create log directory
os.system('mkdir -p logs')
os.system('rm -rf logs/*')

# make the file
ret = os.system('cd build && cmake ../ && make')
assert ret == 0, "make fails"

for i, (scene_name, particle_num, space_size, iteration) in enumerate(scenes):
    k = particle_num // 500
    n_queries = particle_num // 500
    for j, worker in enumerate(workers):
        print(f'--- running {scene_name} on {worker} workers ---')
        init_file = f'benchmark-files/{scene_name}-init.txt'
        query_file = f'benchmark-files/{scene_name}-query.txt'
        if version == 0:
            output_file = f'benchmark-files/{scene_name}-ref.txt'
            os.system(f'rm -rf {output_file}')
            os.system(f'rm -rf {query_file}')
        else:
            output_file = f'logs/{scene_name}.txt'
        log_file = f'logs/{scene_name}.log'
        cmd = f'{prog} -n {particle_num} -i {iteration} -in {init_file} -s {space_size} -o {output_file} -q {query_file} -nq {n_queries} -k {k}'
        # cmd = f'{prog} -n {particle_num} -i {iteration} -in {init_file} -s {space_size} -o {output_file} -nq {n_queries}'
        if version != 0:
            os.environ["OMP_NUM_THREADS"] = str(worker)
            print(os.environ["OMP_NUM_THREADS"])
            # cmd = f" OMP_NUM_THREADS={worker} " + cmd
            cmd += " -p"
        # cmd = perf_prefix + cmd
        cmd += f' > {log_file}'
        # cmd = f'mpirun -n {worker} {prog} {load_balance} -n {particle_num} -i {iteration} -in {init_file} -s {space_size} -o {output_file} > {log_file}'
        ret = os.system(cmd)
        assert ret == 0, 'ERROR -- kdtree exited with errors'
        if version != 0:
            compare(output_file, f'benchmark-files/{scene_name}-ref.txt', k)
        build_time = float(re.findall(
            r'build time: (.*?)s', open(log_file).read())[0])
        traverse_time = float(re.findall(
            r'traverse time: (.*?)s', open(log_file).read())[0])
        print(f'build time: {build_time:.6f}s, traverse time: {traverse_time:.6f}s')
        perfs[i][j][0] = build_time
        perfs[i][j][1] = traverse_time
    
if version == 0:
    for perf in perfs:
        perf_str = ', '.join(f'{x}' for x in perf[0])
        print(f"    ({perf_str}),")
    exit(0)

# print(perfs.shape)

build_speedup = cal_speedup(actual=perfs[:,:,0], ref=ref_perfs[:,0])
print_table("-- Build Tree Performance Table ---", build_speedup)

traverse_speedup = cal_speedup(actual=perfs[:,:,1], ref=ref_perfs[:,1])
print_table("-- Traverse Tree Performance Table ---", traverse_speedup)

timestamp = time.strftime("%m%d-%H%M", time.localtime()) 
sign = f"{timestamp}"
np.savetxt(f"./data/build_speedup-{sign}.csv", build_speedup, fmt="%.2f", delimiter=',')
np.savetxt(f"./data/traverse_speedup-{sign}.csv", traverse_speedup, fmt="%.2f", delimiter=',')
# np.savetxt(f"./data/scenes.csv", scenes, delimiter=',')
# perfs = np.array(perfs)
# print('\n-- Build Tree Performance Table ---')
# header = '|'.join(f' {x:<15} ' for x in ['Scene Name'] + workers)
# print(header)
# print('-' * len(header))
# build_speedup = cal_speedup(actual=perfs[:,:,0], ref=ref_perfs[:,0])
# for scene, perf in zip(scenes, build_speedup):
#     perf_str = '|'.join(f' {x:<15} ' for x in perf)
#     print(f' {scene[0]:<15} |{perf_str}')

# print('\n-- Traverse Tree Performance Table ---')
# header = '|'.join(f' {x:<15} ' for x in ['Scene Name'] + workers)
# print(header)
# print('-' * len(header))
# traverse_speedup = cal_speedup(actual=perfs[:,:,1], ref=ref_perfs[:,1])
# for scene, perf in zip(scenes, traverse_speedup):
#     perf_str = '|'.join(f' {x:<15} ' for x in perf)
#     print(f' {scene[0]:<15} |{perf_str}')

# score = 0.0
# print('\n-- Score Table ---')
# print(header)
# print('-' * len(header))
# for i, (scene, perf) in enumerate(zip(scenes, perfs)):
#     scores = [compute_score(perf[j], ref)
#               for j, ref in enumerate(ref_perfs[i])]
#     score += sum(scores)
#     print('|'.join(f' {x:<15} ' for x in [scene[0]] + scores))
# print(f'total score: {score}/{len(workers) * len(scenes)}')
