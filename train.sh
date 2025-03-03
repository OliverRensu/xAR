
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NPROC=$ARNOLD_WORKER_GPU
export NPROC=$ARNOLD_WORKER_GPU

echo "per worker GPU," $NPROC


torchrun --nproc_per_node=8 --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT \
main_mar.py \
--img_size 256 --vae_path ../../MAR/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model xar_base \
--epochs 800 --warmup_epochs 100 --batch_size 32 --blr 5e-5 \
--output_dir ./output_dir/ --resume ./output_dir/ \
--data_path /mnt/bn/qihangyu-arnold-dataset-eu/imagenet1k/ --use_cached --cached_path /tmp/rsc/DataSet/imagenet_mar_feature/

