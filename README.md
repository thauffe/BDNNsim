# BDNNsim

BDNNsim is a program to simulate a fossil records. Taxa are generated under a birth-death process that can be time-, trait-, diversity-, and paleoenvironment-dependent. Fossil sampling 

## Installation

First, make sure **Python (v.3.11 or higher)** is installed on your computer. To install or upgrade Python visit: [python.org](https://www.python.org/downloads/).
You can run BDNNsim within a virtual environment to make sure all the compatible dependencies are included without affecting your system Python installation. Follow the instructions from the [PyRate tutorial](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_0.md)

After activating the virtual environment, you can install BDNNsim and all its dependencies using git:

```
pip install git+https://github.com/thauffe/BDNNsim.git
```

Or you can download the JaxRate repository, unpack it, and install it from its local directory:

```
pip install /path/to/BDNNsim-main
```

## Usage

BDNNsim can be used form the command line with limited options or it can be loaded as python package with full functionality.

### Command line usage

Check the available options.

```
BDNNsim --help
```

When two numbers are required, they define the range from where a random value is draww. For instance, `-lam 0.1 0.2` means that the birth rate will be taken from this range.


Example of a simple constant birth-death process aiming to generate 200-300 taxa over 30 million years with two continuous and three categorical traits. Fossil sampling will be simulated according to a time-variable Poisson process with heterogeneity among taxa.

```
BDNNsim  -taxa 200 300 -root 30.0 30.0 -cont_traits 2 2 -cat_traits 3 3 -wd /path/to/directory -name sim -q_fixed 0.5 3.0 1.0 2.0 -q_shift 23.03 15.97 2.58 -alpha 1.0 2.0 -seed 242
```

