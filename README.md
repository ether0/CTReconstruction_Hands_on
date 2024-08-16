# CTReconstruction_Hands_on

This repository contains a modified version of the CT reconstruction code originally developed by [leehoy](https://github.com/leehoy/CTReconstruction.git). The primary focus of this project is to improve and adapt the code for specific use cases. Additionally, the phantom data used in this project is generated using the code from [tsadakane's SL3D repository](https://github.com/tsadakane/sl3d.git).

## Overview

This project implements a CT (Computed Tomography) reconstruction algorithm with a simple hands-on projects.

## Requirements

To run the code, ensure you have the following dependencies installed:

- **CUDA** (for GPU acceleration, verified for 11.7)
- **Python 3.10**

The required Python packages can be installed using pip:

```bash
pip install matplotlib==3.9.2 numba==0.60.0 numpy==2.0.1 pycuda==2024.1.2 scipy==1.14.0
```

## Installation

1. Clone this repository
```
git clone https://github.com/ether0/CTReconstruction_Hands_on.git
cd CTReconstruction_Hands_on
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Ensure you have CUDA properly installed and configured on your system.

## Acknowledgments

This project builds upon the work of [leehoy](https://github.com/leehoy/CTReconstruction.git) for the CT reconstruction algorithm and [tsadakane](https://github.com/tsadakane/sl3d.git) for the SL3D phantom generation code.

## License

Please refer to the original repositories for the respective licenses under which the original codes are distributed.
