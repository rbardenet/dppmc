# dppmc

This package implements sampling from multivariate orthonormal polynomial ensembles, as used in the paper

Monte Carlo with determinantal point processes, R. Bardenet and A. Hardy, [https://arxiv.org/abs/1605.00361]

## Installation instructions
To install python dependencies, you can run
```
pip install XXX
```
Then you need to compile a shared library using

gcc -shared -o myJacobiPoly.so myJacobiPoly.c

You should now be able to run the jupyter notebook examples.ipynb in dppmc/docs
