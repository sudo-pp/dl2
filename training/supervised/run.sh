#!/bin/bash


# CIFAR DL2
python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir reports
# python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0.2 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir reports --num-iters 10

# CIFAR Baselines
python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityT(delta=0.01)" --report-dir reports
# python main.py --batch-size 128 --num-epochs 200 --dl2-weight 0 --dataset cifar10  --constraint "CSimilarityG(delta=1.0, eps=0.3)" --report-dir reports --num-iters 10

python results.py --folder reports
