#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5 python basicsr/train.py -opt /home/apinyol/TFM/TFM/Restormer/options/Pretraining_ISR_compiled.yml 