#SBATCH -J deit_fullprecision
#SBATCH -o ./training_logs/%j.log_deit_fullprecision.txt
#SBATCH -p A6000
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=1

WANDB_API_KEY="51ecf17c20b4f4072bad7ef1312b7b7843ea6841" 
export WANDB_API_KEY
wandb login $WANDB_API_KEY

DIR=output
VERSION=fullprecision_300

mkdir -p ${DIR}/${VERSION}

nohup python3 ./main_original.py \
--model deit_small_patch16_224 \
--epochs 300 \
--weight-decay 0.05 \
--batch-size 256 \
--data-path /home/tsyeom/dataset/imagenet \
--lr 5e-4 \
--output_dir ${DIR}/${VERSION} \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &






