#!/bin/sh

#SBATCH -J deit_base_lbt
#SBATCH -o ./output
#SBATCH -t 72:00:00
#### Select GPU
#SBATCH -p RTX6000ADA
#SBATCH --gres=gpu:4
## node 지정하기
#SBATCH --nodelist=n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"
srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

## Python Virtual Env ## ⇒> 가상환경
echo "Start"
echo "conda PATH "
echo "source home/tsyeom/anaconda3/etc/profile.d/conda.sh"
source home/tsyeom/anaconda3/etc/profile.d/conda.sh #경로
echo "conda activate deit2 "
conda activate deit2 #사용할 conda env

SAMPLES_DIR=$HOME/deit

DIR=output
VERSION=deit_base_finegrained_wgfp_qkl_vertex_register_token_rand_init

mkdir -p ${DIR}/${VERSION}

python -u $SAMPLES_DIR/main_original.py \
--model fourbits_deit_base_patch16_224 \
--epochs 300 \
--weight-decay 0.05 \
--batch-size 256 \
--data-path /home/tsyeom/dataset/imagenet \
--lr 5e-4 \
--output_dir ${DIR}/${VERSION} \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &

date
echo " conda deactivate deit2 "
conda deactivae # 마무리 deactivate
squeue --job $SLURM_JOBID
echo "##### END #####"






