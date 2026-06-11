#!/bin/bash
#SBATCH --job-name=embeds
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

# MODEL_SIZE=$1  # Pass L or XXL
# DATASET=$2     # Pass beauty, toys, or sports

# # Force the input to uppercase just in case
# MODEL_SIZE=$(echo "$1" | tr '[:lower:]' '[:upper:]')
# DATASET=$2

# if [[ -z "$MODEL_SIZE" || -z "$DATASET" ]]; then
#     echo "Usage: sbatch embedding.sh [L|XXL] [beauty|toys|sports]"
#     exit 1
# fi

# # Case-insensitive mapping block
# if [[ "$MODEL_SIZE" == "L" || "$MODEL_SIZE" == "LARGE" ]]; then
#     HF_MODEL="large"
# elif [[ "$MODEL_SIZE" == "XXL" ]]; then
#     HF_MODEL="xxl"
# else
#     echo "Invalid model size: $MODEL_SIZE. Choose L or XXL."
#     exit 1
# fi

cd $HOME/GRID

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RecSys

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export OMP_NUM_THREADS=8

BASE=/projects/prjs2120/groups/group_08

# python -m src.inference \
#     experiment=sem_embeds_inference_flat \
#     data_dir=$BASE/data/amazon_data/$DATASET \
#     hydra.run.dir=$BASE/results/embeddings/${MODEL_SIZE}/$DATASET \
#     embedding_model=google/flan-t5-${HF_MODEL}

python -m src.inference \
    experiment=sem_embeds_inference_flat \
    data_dir=/projects/prjs2120/groups/group_08/data/amazon_data/beauty \
    hydra.run.dir=/projects/prjs2120/groups/group_08/results/embeddings/L/beauty \
    embedding_model=google/flan-t5-large