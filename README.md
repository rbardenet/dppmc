# dppmc

This package implements sampling from multivariate orthonormal polynomial ensembles, as used in the paper

*Monte Carlo with determinantal point processes*, R. Bardenet and A. Hardy, [https://arxiv.org/abs/1605.00361]

## Installation instructions
Our instructions cover Unix systems. To install python dependencies, you can run
```
pip install numpy==1.10.4 scipy==0.17.0 matplotlib==1.5.1 seaborn==0.7.1 pandas==0.20.2 progressbar2==3.30.2
```
Then you need to compile a shared C library using
```
gcc -shared -o myJacobiPoly.so myJacobiPoly.c
```
You should now be able to run the examples
```
cd dppmc/examples
python example1.py
```
