#!/bin/bash

conda env create -p ../env/ -f ../env.yml
conda activate ../env/
python -m ipykernel install --user --name=env
conda deactivate
