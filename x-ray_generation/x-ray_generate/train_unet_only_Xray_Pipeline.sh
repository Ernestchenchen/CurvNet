#https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
export MODEL_NAME="/data1/zhiwen/stable-diffusion-2-1-base/"
#export MODEL_NAME="/mnt/data/experiments/stable-diffusion-v1-5/"
export INSTANCE_DIR="/data1/zhiwen/datasets/STB_augmentation"
export OUTPUT_DIR="../pretraining_unet_dataaugmentation_1"



accelerate launch train_unet_only_Xray_Pipeline.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="X-ray of scoliosis" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 
