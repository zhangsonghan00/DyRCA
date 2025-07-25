#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-padi-rca}
LOGURU_COLORIZE=0 .venv/bin/python run_exp.py