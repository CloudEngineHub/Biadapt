CUDA_VISIBLE_DEVICES=1 python train/train_affordance_sec.py \
    --exp_suffix 0910_50 \
    --category_types Pliers  \
    --primact_type opening \
    --model_version model_aff_sec \
    --actor_version model_actor_sec   \
    --actor_path   ../logs/actor/exp-model_actor_sec-opening-Pliers-0910_50 \
    --actor_eval_epoch  80 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-model_critic_sec-opening-Pliers-0910_50 \
    --critic_eval_epoch 80 \
    --offline_data_dir  ../exp/Pliers_Opening \
    --val_data_dir   ../exp/Pliers_Opening \
    --train_buffer_max_num 50  \
    --val_buffer_max_num 20 \
    --feat_dim 128   \
    --batch_size 8  \
    --lr 0.001      \
    --lr_decay_every 300 \
    --succ_proportion 0.5 \
    --fail_proportion 0.5 \
    --z_dim 32      \
    --topk  100       \
    --coordinate_system world \
    --mode all \
    --exchange_ctpts
    


