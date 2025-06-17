import torch
import gc
import platform
import shutil
import subprocess

class Colour: 
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def check_gpu_memory(mem_threshold_gb=1.6):
    use_gpu = False
    device = torch.device('cpu')

    if torch.cuda.is_available():
        if shutil.which("nvidia-smi") is not None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                    encoding='utf-8',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                free_memory_mb = int(result.stdout.strip().split('\n')[0])  # First GPU
                free_memory_gb = free_memory_mb / 1024

                if free_memory_gb >= mem_threshold_gb:
                    use_gpu = True
                    device = torch.device('cuda:0')
                    print(f"Using GPU with {free_memory_gb:.2f} GB memory available")
                else:
                    print(f"Only {free_memory_gb:.2f} GB GPU memory available, switching to CPU.")

            except Exception as e:
                print(f"Error checking GPU memory with nvidia-smi: {e}")
                print("Defaulting to CPU.")
        else:
            print("nvidia-smi not found. GPU memory check skipped, using CPU.")
    else:
        print("CUDA not available, using CPU.")

    return device, use_gpu


def test_cuda(mem_threshold_gb=2.0):
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    if platform.system() == 'Windows' and gpu:
        gc.collect()
        torch.cuda.empty_cache()
        device, gpu = check_gpu_memory(mem_threshold_gb=mem_threshold_gb)
    return device, gpu

if __name__ == '__main__':
    device, gpu = test_cuda()
    print(f'Using: {device}')