export OMP_NUM_THREADS=1


WANDB_API_KEY="51ecf17c20b4f4072bad7ef1312b7b7843ea6841" 
export WANDB_API_KEY
wandb login $WANDB_API_KEY

DIR=output
VERSION=fullprecision_300

mkdir -p ${DIR}/${VERSION}

python3 ./main_original.py \
--model deit_small_patch16_224 \
--epochs 300 \
--weight-decay 0.05 \
--batch-size 256 \
--data-path /home/tsyeom/dataset/imagenet \
--lr 5e-4 \
--output_dir ${DIR}/${VERSION} \
--distributed > ${DIR}/${VERSION}/output.log 2>&1 &






