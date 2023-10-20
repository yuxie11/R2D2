export NUM_GPUS=8
export NUM_NODES=1

export MASTER_ADDR=localhost
export MASTER_PORT=23439


python -m torch.distributed.run \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$NUM_GPUS" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  finetune_r2d2_matching.py --config config/matching_iqm_large.yaml 
