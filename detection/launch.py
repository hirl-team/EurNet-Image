#!/usr/bin/python3
import os
import sys
import socket
import random
import argparse
import subprocess
import time
import torch

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def _get_rand_port():
    random.seed()
    return random.randrange(20000, 60000)

def init_workdir():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Launcher')
    parser.add_argument('--np', type=int, default=-1,
                        help='number of processes per node')
    parser.add_argument('--nn', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--port', type=int, default=-1,
                        help='master port for communication')
    parser.add_argument('--nr', type=int, default=0, 
                        help='node rank.')
    parser.add_argument('--no-local-log', action='store_true',
                        help='disable local logging')
    parser.add_argument('--master_address', '-ma', type=str, default="127.0.0.1")
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--output_dir', type=str, default='./object_detection/coco/eurnet_base_default/')
    parser.add_argument('--config', '-c', type=str, default='./configs/mask_rcnn/mask_rcnn_eurnet_base_1x_coco.py')

    args, other_args = parser.parse_known_args()

    init_workdir()

    master_address = args.master_address
    num_processes_per_worker = torch.cuda.device_count() if args.np < 0 else args.np
    num_workers = args.nn
    node_rank = args.nr

    # Get port
    if args.port > 0:
        master_port = args.port
    elif num_workers == 1:
        master_port = _find_free_port()
    else: 
        master_port = _get_rand_port()


    args.output_dir = os.path.normpath(args.output_dir)
    output_dir_prefix = os.path.dirname(args.output_dir)

    os.makedirs(output_dir_prefix, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    
    start_time = time.time()

    # start training
    print(f'Start training on coco by torch.distributed.launch with port {master_port}!', flush=True)
    os.environ['NPROC_PER_NODE'] = str(num_processes_per_worker)
    cmd = f'python3 -m torch.distributed.launch \
            --nproc_per_node={num_processes_per_worker} \
            --nnodes={num_workers} \
            --node_rank={node_rank} \
            --master_addr={master_address} \
            --master_port={master_port} \
            train.py \
            --launcher pytorch \
            {args.config} \
            --work-dir {args.output_dir} \
            --deterministic \
            --cfg-options model.pretrained={args.pretrained}'
    for argv in other_args:
        cmd += f' {argv}'
    if args.no_local_log is False:
        with open(os.path.join(args.output_dir, 'log_coco_train.txt'), 'wb') as f:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                text = proc.stdout.readline()
                f.write(text)
                f.flush()
                sys.stdout.buffer.write(text)
                sys.stdout.buffer.flush()
                exit_code = proc.poll()
                end_time = time.time()
                if exit_code is not None:
                    break
    else:
        exit_code = subprocess.call(cmd, shell=True)

    # start testing
    print(f'Start testing on coco by torch.distributed.launch with port {master_port}!', flush=True)

    cmd = f'python3 -m torch.distributed.launch \
            --nproc_per_node={num_processes_per_worker} \
            --nnodes={num_workers} \
            --node_rank={node_rank} \
            --master_addr={master_address} \
            --master_port={master_port} \
            ./test.py \
            --gpu-collect \
            --launcher pytorch \
            {args.config} \
            {os.path.join(args.output_dir, "latest.pth")} \
            --eval bbox segm'
    if other_args:
        cmd += ' --cfg-options'
        for argv in other_args:
            cmd += f' {argv}'

    if args.no_local_log is False:
        with open(os.path.join(args.output_dir, 'log_coco_test.txt'), 'wb') as f:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                text = proc.stdout.readline()
                f.write(text)
                f.flush()
                sys.stdout.buffer.write(text)
                sys.stdout.buffer.flush()
                exit_code = proc.poll()
                if exit_code is not None:
                    break
    else:
        exit_code = subprocess.call(cmd, shell=True)
    sys.exit(exit_code)
