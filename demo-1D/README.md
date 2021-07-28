## R-sens demo for functions with 1 predictor variable

This demo plots the R-sens uncertainty-aware sensitivity measure for user-defined functions. In `demo-gaussian.py`, you can define the mean and variance of a Gaussian distribution as a function of `x`. In `demo-bernoulli.py`, you can define the classification probability of a Bernoulli distribution as a function of `x`.

## Running the demo

Run the demo with Python simply with the command
```
python demo-gaussian.py
```
or
```
python demo-bernoulli.py
```
You can try the demo with different function shapes simply by editing the mean (`E`) and variance (`V`) functions in `demo-gaussian.py` or the classification probability function in `demo-bernoulli.py`.
