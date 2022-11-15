#!/usr/bin/python3
import os
import sys
import socket
import random
import argparse
import subprocess
import time
import glob
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
    parser = argparse.ArgumentParser(description='Segmentation Launcher')
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
    parser.add_argument('--output_dir', type=str, default='./segmentation/ade20k/eurnet_base_default/')
    parser.add_argument('--config', '-c', type=str, default='./configs/eurnet/upernet_eurnet_base_512x512_160k_ade20k.py')

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

    ## manual resume from last
    pth_list = glob.glob(os.path.join(args.output_dir, "iter_*.pth"))
    last_path_file = None
    if len(pth_list):
        pth_list.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("iter_")[-1]))
        last_path_file = pth_list[-1]
    else:
        last_path_file = None
    
    
    start_time = time.time()

    # start training
    print(f'Start training on ADE20K by torch.distributed.launch with port {master_port}!', flush=True)
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
            --deterministic '
    if last_path_file:
        print("training resume from {}".format(last_path_file))
        cmd+=f'--resume-from {last_path_file} '
    cmd+=f'--options model.pretrained={args.pretrained}'
    for argv in other_args:
        cmd += f' {argv}'

    if args.no_local_log is False:
        with open(os.path.join(args.output_dir, 'log_ADE20K_train.txt'), 'ab') as f:
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

    pth_list = glob.glob(os.path.join(args.output_dir, "iter_*.pth"))
    if len(pth_list):
        pth_list.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("iter_")[-1]))
        last_path_file = pth_list[-1]
    else:
        last_path_file = None
    # start testing
    print(f'Start testing on ADE20K by torch.distributed.launch with port {master_port}!', flush=True)

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
            {last_path_file} \
            --eval mIoU --aug-test'
    if other_args:
        cmd += ' --options'
    for argv in other_args:
        cmd += f' {argv}'

    if args.no_local_log is False:
        with open(os.path.join(args.output_dir, 'log_ADE20K_test.txt'), 'ab') as f:
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
