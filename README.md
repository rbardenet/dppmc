# dppmc

This package implements sampling from multivariate Jacobi ensembles, as used in the paper

*Monte Carlo with determinantal point processes*, R. Bardenet and A. Hardy, [https://arxiv.org/abs/1605.00361]

## Prerequirements
Our instructions cover Linux and Mac only. You need the [gcc](https://gcc.gnu.org/) C compiler installed. On Linux, it usually comes preinstalled. On Mac, you can install it as part of the [XCode command line tools](https://developer.apple.com/xcode/).

## Install from sources
Start by cloning this repository
```
git clone https://github.com/rbardenet/dppmc.git
cd dppmc
```
then run `setup.py`, e.g. using `pip`
```
pip install .
```
and compile the following shared C library
```
gcc -shared -o myJacobiPoly.so myJacobiPoly.c
```
You should now be able to run the example
```
cd dppmc/examples
python example1.py
```
Such examples, along with various illustrations of the notions in the paper, like the graded lexicographic order or the bound used in rejection sampling, can be found in a Jupyter notebook
```
cd dppmc/docs
jupyter notebook examples.ipynb
```
