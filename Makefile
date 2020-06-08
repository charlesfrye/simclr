# Use:
#
# `make container` will build a container
#

IMAGE_PARENT ?=tensorflow/tensorflow:1.15.0-gpu-py3
IMAGE_NAME ?=simclr
WANDB_IMAGE ?=simclr:wandb
WANDB_PROJECT ?=simclr

container:
	docker build -t $(IMAGE_NAME) --build-arg PARENT=${IMAGE_PARENT} .

# pretrain or finetune docker on a single GPU
simclr:
	docker run -it -u `id -u`:`id -g` -v ${DATA_DIR}:/data --runtime=nvidia \
	--shm-size=1g --ulimit memlock=-1 -e CUDA_VISIBLE_DEVICES=0 \
	${IMAGE_NAME} /bin/bash

# inside the docker

# make -n run_pretrain DATA_DIR=cifar10 MODEL_DIR=/data/cifar10_model
run_pretrain:
	python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5 \
  --dataset=cifar10 --data_dir=${DATA_DIR} \
  --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=${MODEL_DIR} --use_tpu=False

# make -n run_finetune DATA_DIR=cifar10 CHECKPOINT=/data/cifar10_model MODEL_DIR=/data/cifar10_model_ft
run_finetune:
	python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --image_size=32 --eval_split=test --resnet_depth=18 \
  --dataset=cifar10 --data_dir=${DATA_DIR} \
  --checkpoint=${CHECKPOINT} --model_dir=${MODEL_DIR} --use_tpu=False

# create WANDB sweep for hyper-parameter tuning
wandb_container:
	docker build -t $(WANDB_IMAGE) --build-arg PARENT=${IMAGE_PARENT} \
	--build-arg WANDB_BASE_URL=${WANDB_BASE_URL} -f Dockerfile.wandb .

# make create_sweep WANDB_DIR=... WANDB_USERNAME=... WANDB_API_KEY=... WANDB_PROJECT=simclr SWEEP_CONFIG=...
create_sweep: wandb_container
	mkdir -p ${WANDB_DIR}/wandb_sweep ${WANDB_DIR}/ckpt_sweep
	docker run -u `id -u`:`id -g` -v ${WANDB_DIR}:/data -w /data/wandb_sweep \
	-e WANDB_CONFIG_DIR=/data/wandb_sweep -e WANDB_USERNAME=${WANDB_USERNAME} \
	-e WANDB_API_KEY=${WANDB_API_KEY} -e WANDB_PROJECT=${WANDB_PROJECT} \
	${WANDB_IMAGE} wandb sweep /data/${SWEEP_CONFIG}

# make start_sweep WANDB_DIR=... WANDB_USERNAME=... WANDB_API_KEY=... WANDB_PROJECT=simclr SWEEP_ID=...
start_sweep: wandb_container
	mkdir -p ${WANDB_DIR}/wandb_sweep ${WANDB_DIR}/ckpt_sweep
	docker run -d --runtime=nvidia -u `id -u`:`id -g` -v ${WANDB_DIR}:/data \
	--shm-size=1g --ulimit memlock=-1 -e CUDA_VISIBLE_DEVICES=0 \
	-w /data/wandb_sweep -e WANDB_CONFIG_DIR=/data/wandb_sweep \
	-e WANDB_USERNAME=${WANDB_USERNAME} -e WANDB_API_KEY=${WANDB_API_KEY} -e WANDB_PROJECT=${WANDB_PROJECT} \
	${IMAGE_NAME} wandb agent ${WANDB_OPTION} ${SWEEP_ID}
