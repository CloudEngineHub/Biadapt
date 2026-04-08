CUDA_VISIBLE_DEVICES=2 python train/train_actor_sec.py \
    --exp_suffix 0826 \
    --category_types  Pen \
    --model_version model_actor_sec  \
    --primact_type uncapping \
    --offline_data_dir  ../exp/Pen_Uncapping \
    --val_data_dir   ../exp/Pen_Uncapping \
    --train_buffer_max_num 800  \
    --val_buffer_max_num 100  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001 \
    --lr_decay_every 200 \
    --lbd_kl 0.02     \
    --lbd_dir 5.0     \
    --z_dim 32   \
    --coordinate_system world \
    --mode all \
    --exchange_ctpts


