CUDA_VISIBLE_DEVICES=2 python train/train_actor_sec.py \
    --exp_suffix 0910_50 \
    --category_types  Pliers \
    --model_version model_actor_sec \
    --primact_type opening \
    --offline_data_dir  ../exp/Pliers_Opening \
    --val_data_dir  ../exp/Pliers_Opening \
    --train_buffer_max_num 50  \
    --val_buffer_max_num 20  \
    --feat_dim 128   \
    --batch_size 8  \
    --lr 0.001 \
    --lr_decay_every 200 \
    --lbd_kl 0.02     \
    --lbd_dir 5.0     \
    --z_dim 32   \
    --coordinate_system world \
    --mode all \
    --exchange_ctpts


