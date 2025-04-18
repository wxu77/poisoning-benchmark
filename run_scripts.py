import os
import argparse
from multiprocessing import Queue, Process
import subprocess


def worker(q: Queue, gpu: int):
    while q.qsize() > 0:
        try:
            script_path = q.get(timeout=1)
            with open(script_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if len(line.strip()) > 0:
                    command = f'CUDA_VISIBLE_DEVICES={gpu} {line.strip()}'
                    print(f"running {command}")
                    subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(e)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="/home/y485lu/William/data/scripts/transfer_attack_scripts")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--max_in_parallel", type=int, default=5)
    parser.add_argument("--sub_folders", type=str, nargs="+")

    args = parser.parse_args()

    q = Queue()
    for folder in args.sub_folders:
        folder_path = os.path.join(args.folder, folder)
        assert os.path.isdir(folder_path), folder_path
        for script in os.listdir(folder_path):
            if script.endswith(".sh"):
                q.put(os.path.join(folder_path, script))
    
    procs = []
    for i in range(args.max_in_parallel):
        proc = Process(target=worker, args=(q, i % args.num_gpus))
        procs.append(proc)
    
    for proc in procs:
        proc.start()
    
    for proc in procs:
        proc.join()
