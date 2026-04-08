CUDA_VISIBLE_DEVICES=2 python Correspondence/mapctpts.py \
    --method dift \
    --dataset clip_b32 \
    --base_dir ../exp/Retrieve_Pliers_0103 \
    --out_dir ../exp/Cor_res/Pliers_0104 \
    --top_k 10
    # --visualize

