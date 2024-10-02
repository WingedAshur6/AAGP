# An Adjacency-Adaptive Gaussian Process Method for Sample Efficient Response Surface Modeling and Test-Point Acquisition
By Stanford Martinez and Adel Alaeddini


## Intro
This repository contains the code for running two examples of simulations as seen in the manuscript (titled above). The specific examples are for the `(1) Qing (3D)` and `(2) Cosine (10D)` functions in Figure 3 therein.

## Recommended Installation
### NOTE
- The steps below are listed to allow users to configure an environment similar to the one used by the authors of this manuscript to produce examples of output as seen in Figure 3 in the manuscript.
- This is our prescribed method of running this code (tested on `Windows`). We recommend installing a separate version of python (versions 3.7-3.9) to use for this example to prevent overwriting any packages currently installed on existing versions of python you may have.
- We have tested on python versions 3.7-3.12 and have found that 3.7-3.9 are compatible with the Deep Gaussian Process model used as a comparison framework (Github: https://github.com/SheffieldML/PyDeepGP)
  - The DeepGP package implemented therein utilizes earlier versions than 3.7 and leverages the `GPy` package, but we have found that the two are both compatible with up to 3.9 before C++ compiler issues arise.
  - For specifics on python versions referenced by the authors of the DeepGP package, please see line 37 here (other specifics are available in this file): https://github.com/SheffieldML/PyDeepGP/blob/master/setup.py#L37

### Setup and Running
1) Download this branch (`main`) as a zip file
2) Extract all contents into some folder `example: C:\Users\your user name\downloads\aagp_demo`
3) install anaconda and run the `anaconda PowerShell prompt` application, then enter the following commands:
4) `conda create --name aagp_demo python=3.9 -y`
5) `conda install -y git`
6) `conda activate aagp_demo`
7) `cd "C:\Users\your user name\downloads\aagp_demo"`
8) `python EXECUTOR.py`

### Changing the Example to Test
1) open the `EXAMPLE_FUNCTION.py` file
2) Change the value of `test_function`:
  - `0` to run a simulation for the `Qing (3D)` function
  - `1` to run a simulation for the `Cosine (10D)` function

## Simulation Runtime Notes
- Once started after step `(8)` above, the code will then proceed to install packages and run the simulation.
- At the end, it will output `Example Output.jpg` to the directory in which `EXECUTOR.py` is placed.
- The DeepGP methodolgy is very memory-intensive and may cause OOM errors running in parallel.
- Runtime is roughly ~100min on an intel 13900HX processor with 28 cores and 96GB RAM running in parallel for the `Qing (3D)` example, and ~360min for the `Cosine (10D)` Example.
- This large difference in compute time reported for the `Cosine (10D)` function is primarily attributed to the DeepGP competitor, the layers and dimensions used for its configuration, the dimensionality of the dataset, and 2,000 iterations (as is used by the model in our manuscript) for DeepGP's training.
- The `Qing (3D)` Example takes a shorter amount of time due to similar results as seen in the manuscript being achievable with using only 200 training iterations instead of 2,000.

