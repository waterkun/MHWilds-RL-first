import torch

def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. Training will be very slow on CPU.")

if __name__ == "__main__":
    check_gpu()