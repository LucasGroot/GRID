#!/bin/bash
#SBATCH --job-name=infer_sids
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

MODEL_SIZE=$1
DATASET=$2
CKPT_NAME=$3

if [[ -z "$MODEL_SIZE" || -z "$DATASET" || -z "$CKPT_NAME" ]]; then
    echo "Usage: sbatch rkmeans_inference.sh [L|XXL] [beauty|toys|sports] [checkpoint_name.ckpt]"
    exit 1
fi

if [ "$MODEL_SIZE" = "L" ]; then
    DIM=1024
elif [ "$MODEL_SIZE" = "XXL" ]; then
    DIM=4096
else
    echo "Invalid model size. Choose L or XXL."
    exit 1
fi

module purge
module load 2025
module load Anaconda3/2025.06-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RecSys

cd $HOME/GRID
export OMP_NUM_THREADS=8
BASE=/projects/prjs2120/groups/group_08

mkdir -p $BASE/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_inference/metadata

python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=$BASE/data/amazon_data/$DATASET \
    embedding_path=$BASE/results/embeddings/${MODEL_SIZE}/${DATASET}/pickle/merged_predictions_tensor.pt \
    embedding_dim=$DIM \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=$BASE/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_train/checkpoints/$CKPT_NAME \
    hydra.run.dir=$BASE/results/sid_rkmeans/${MODEL_SIZE}/${DATASET}/rkmeans_inference