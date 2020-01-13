#!/bin/bash

CURRENT_DIR=$(pwd)

export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR/..

python bin/train.py
