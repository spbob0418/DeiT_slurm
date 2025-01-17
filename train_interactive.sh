#!/bin/sh

#SBATCH -J deit_base_lbt
#SBATCH -o ./%j.log_deit_base_vertex_register.txt
#SBATCH -t 72:00:00
#### Select GPU
#SBATCH -q hpgpu
#SBATCH -p A100-80GB
#SBATCH --gres=gpu:4
## node 지정하기
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"
srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

## Python Virtual Env ## ⇒> 가상환경
echo "Start"
echo "conda PATH "
echo "source /home/tsyeom/anaconda3/etc/profile.d/conda.sh" 
echo "conda activate deit2 "
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deit2 

SAMPLES_DIR=$HOME/deit

DIR=/home/tsyeom/deit/output
VERSION=deit_base_qkl_wgpf_int4_finegrained
export DIR VERSION 
mkdir -p $DIR/$VERSION

cd $DIR/$VERSION 

python -u $SAMPLES_DIR/main_original.py \
--model fourbits_deit_base_patch16_224 \
--epochs 90 \
--weight-decay 0.05 \
--batch-size 256 \
--data-path /home/tsyeom/dataset/imagenet \
--lr 5e-4 \
--output_dir $DIR/$VERSION \
--distributed > $DIR/$VERSION/output.log 2>&1

date
echo " conda deactivate deit2 "
conda deactivate # 마무리 deactivate
squeue --job $SLURM_JOBID
echo "##### END #####"






