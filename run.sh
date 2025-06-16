#!/bin/bash
#SBATCH --job-name=srihari                # Job name
#SBATCH --output=srihari.txt              # Output file
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --time=59:59                  # Time limit hh:mm:ss
#SBATCH --mem=120G                        # Memory limit

#SBATCH --gres=gpu:1                      # Request one GPU
#SBATCH --nodelist=node2                  # Use node2 (H100)

# Load CUDA
module load cuda/cuda-11.7

# Activate Conda environment
source ~/.bashrc
conda activate vit38

# Run the script
python3 /export/home/srihari/REL_CAM/Transformer-Explainability/demo.py
