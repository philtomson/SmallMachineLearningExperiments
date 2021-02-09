## Small Machine Learning Experiments
This repo inspired by a blog entry by Sam Greydanus entitled *Scaling down Deep Learning* (https://greydanus.github.io/2020/12/01/scaling-down/)

The main idea is to create small ML models that can be trained in a matter of minutes and use those models as testbeds for experimenting with different architectures, activation functions, parameters, etc.

## Testcases

* count_ones.jl: The problem is to find the number of 1's in a vector of length 60 where the number of 1's can vary between 1 and 10. Training and test sets are generated randomly without any duplicates. Using the GPU training takes only about 5 minutes for 20000 training cases and 10000 test cases. One potential flaw in the data is that in a vector of length 60 there are only 60 possible cases with one 1.

