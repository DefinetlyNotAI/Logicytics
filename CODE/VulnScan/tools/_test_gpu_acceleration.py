try:
    # noinspection PyUnresolvedReferences
    import torch
except ImportError as e:
    print(f"Error: Failed to import torch. Please ensure PyTorch is installed correctly: {e}")
    exit(1)


def check_gpu():
    """Check if CUDA is available and print the device information.

        This function attempts to detect CUDA capability and prints whether
        GPU acceleration is available, along with the device name if applicable.
    """
    try:
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
    except RuntimeError as err:
        print(f"Error initializing CUDA: {err}")


check_gpu()
