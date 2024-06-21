# Symmetric padding for Pytorch

Welcome to the `sympad_pytorch` repository!

## Description

This repository adds `symmetric` padding to PyTorch. Symmetric padding for example is the default in `pywt` (https://pywavelets.readthedocs.io). Providing this functionality in PyTorch as a C++ module will allow us to speed up Wavelet computations in PyTorch.

## Testing and Verification

Follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/cpp_pad.git`.
2. Navigate to the project directory: `cd sympad_pytorch `.
3. Run the tests with `nox -s test`.


