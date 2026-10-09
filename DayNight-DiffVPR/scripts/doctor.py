"""A fast version/hardware smoke-check. No weights or external downloads needed."""
from __future__ import annotations
import importlib.metadata
import platform
import sys


def main():
    print('Python:', sys.version.split()[0],platform.system(),platform.machine())
    try:
        import torch
        print('torch:',torch.__version__)
        print('CUDA available:',torch.cuda.is_available())
        if torch.cuda.is_available():
            device=torch.cuda.get_device_properties(0)
            print('GPU:',device.name,'memory GiB:',round(device.total_memory/2**30,1))
            print('BF16 supported:',torch.cuda.is_bf16_supported())
    except ImportError as e:
        print('torch missing:',e)
    for name in ('torchvision','diffusers','transformers','peft','accelerate','huggingface-hub','opencv-python-headless','safetensors'):
        try:
            print(f'{name}: {importlib.metadata.version(name)}')
        except importlib.metadata.PackageNotFoundError:
            print(f'{name}: NOT INSTALLED')


if __name__=='__main__':
    main()
