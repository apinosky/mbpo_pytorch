## Overview
This is a re-implementation of the model-based RL algorithm MBPO in pytorch as described in the following paper: [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253).

This code is based on a [previous paper in the NeurIPS reproducibility challenge](https://openreview.net/forum?id=rkezvT9f6r) that reproduces the result with a tensorflow ensemble model but shows a significant drop in performance with a pytorch ensemble model.
This code re-implements the ensemble dynamics model with pytorch and closes the gap.

## Reproduced results
The comparison are done on two tasks while other tasks are not tested. But on the tested two tasks, the pytorch implementation achieves similar performance compared to the official tensorflow code.
![alt text](./results/hopper.png) ![alt text](./results/walker2d.png)
## Dependencies

MuJoCo 1.5 & MuJoCo 2.0

## Usage
> python main_mbpo.py --env_name 'Walker2d-v2' --num_epoch 300 --model_type 'pytorch'

> python main_mbpo.py --env_name 'Hopper-v2' --num_epoch 300 --model_type 'pytorch'

## Reference
* Official tensorflow implementation: https://github.com/JannerM/mbpo
* Code to the reproducibility challenge paper: https://github.com/jxu43/replication-mbpo


## alp mods
- replaced SAC repository with the one we used for hybrid learning (original sac implementation included here took forever to run)
- modified rollout length to a fixed value for comparison purposes
- changed number of networks from 7 > 1 and elites from 5>1 (better match comparisons / run faster)
- changed model sizes to match other comparisons (4 layers > 2 layers), removed weight decay, and changed activation function from swish to relu
- changed training batch sizes from 256 > 128
- added model_train_batch_size to only train model on a subset of the data (rather than all data collected for it to run faster)
- modified rollout_batch_size and num_train_repeat for run time reasons as well
- added jupyter notebook for visualization
- saved config to `requirements.txt` with `pip freeze --local`
- added HalfCheetah env for comparison
- removed termination function for comparison
