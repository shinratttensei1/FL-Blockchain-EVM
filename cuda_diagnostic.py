#!/usr/bin/env python3
"""CUDA diagnostic script to help troubleshoot GPU issues."""

import sys
import subprocess


def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1


def check_cuda_installation():
    """Check CUDA installation."""
    print("=== CUDA Installation Check ===")

    # Check nvidia-smi
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print("✓ nvidia-smi works")
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line or 'CUDA Version' in line:
                print(f"  {line.strip()}")
    else:
        print("✗ nvidia-smi failed")
        print(f"  Error: {stderr}")

    # Check nvcc
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✓ nvcc works")
        for line in stdout.split('\n'):
            if 'release' in line:
                print(f"  {line.strip()}")
    else:
        print("✗ nvcc not found or failed")

    # Check CUDA libraries
    stdout, stderr, code = run_command(
        "ls /usr/local/cuda/lib64/ 2>/dev/null | head -5")
    if code == 0:
        print("✓ CUDA libraries found in /usr/local/cuda/lib64/")
    else:
        print("✗ CUDA libraries not found in standard location")


def check_gpu_architecture():
    """Check GPU architecture compatibility."""
    print("\n=== GPU Architecture Check ===")

    try:
        import torch
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            arch = f"{device_props.major}.{device_props.minor}"
            print(f"GPU Architecture: {arch}")
            print(f"GPU Name: {device_props.name}")

            # Check if architecture is supported by CUDA 12.6
            # CUDA 12.6 supports architectures: 6.0+ (Pascal and newer)
            major, minor = device_props.major, device_props.minor
            if major < 6:
                print("❌ GPU architecture too old for CUDA 12.6")
                print("   Need Pascal (6.0) or newer GPU")
                return False
            else:
                print("✅ GPU architecture supported by CUDA 12.6")
                return True
        else:
            print("No CUDA GPU available")
            return False
    except Exception as e:
        print(f"Error checking GPU architecture: {e}")
        return False


def main():
    """Main diagnostic function."""
    print("CUDA Diagnostic Tool")
    print("=" * 50)

    check_cuda_installation()
    check_gpu_architecture()

    print("\n=== Recommendations ===")
    print("If CUDA test failed:")
    print("1. Check PyTorch installation: pip show torch")
    print("2. Install correct PyTorch version for your CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   (replace cu118 with your CUDA version: cu117, cu121, etc.)")
    print("3. Or install CPU-only version: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("4. Update NVIDIA drivers if needed")
    print("5. If GPU is old (pre-Pascal), you need older PyTorch/CUDA")


if __name__ == "__main__":
    main()
