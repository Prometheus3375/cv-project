### Installation
1. Install [Python 3](https://www.python.org/downloads/).
1. In terminal change current working directory to the root of this repository.
1. (Optional) Initialize virtual environment and activate it according to the
   [tutorial](https://docs.python.org/3/library/venv.html).
1. [Update pip](https://pip.pypa.io/en/stable/installing/#upgrading-pip).
1. Run `pip install -U setuptools wheel`. This will update setuptools and wheel packages.
1. Install necessary drivers and software to allow tensorflow use GPU. Follow this 
   [guide](https://www.tensorflow.org/install/gpu#software_requirements).
1. Run `pip install tensorflow`. This will install tensorflow package.
1. Install PyTorch with CUDA version supported by your GPU according to the 
   [guidelines](https://pytorch.org/).
1. Run `pip install -r requirements.txt`. This will install all necessary packages for the project.
