CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6.7 python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:29040" \
  train_r2d2_retrieval.py  --config ./config/retrieval_flickr_cna_large.yaml --output_dir output/flickr