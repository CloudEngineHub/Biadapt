CUDA_VISIBLE_DEVICES=7 python train/train_affordance_fir.py \
    --exp_suffix 0910_50 \
    --category_types Pliers  \
    --primact_type opening \
    --model_version model_aff_fir \
    --actor_version model_actor_fir  \
    --actor_path  ../logs/actor/exp-model_actor_fir-opening-Pliers-0910_50 \
    --actor_eval_epoch 165 \
    --critic_version model_critic_fir   \
    --critic_path  ../logs/critic/exp-model_critic_fir-opening-Pliers-0910_50 \
    --critic_eval_epoch 120 \
    --offline_data_dir  ../exp/Pliers_Opening \
    --val_data_dir   ../exp/Pliers_Opening \
    --train_buffer_max_num 50  \
    --val_buffer_max_num 20  \
    --feat_dim 128   \
    --batch_size 8  \
    --lr 0.001      \
    --lr_decay_every 200 \
    --succ_proportion 0.5 \
    --fail_proportion 0.5 \
    --z_dim 32      \
    --topk  100       \
    --coordinate_system world \
    --mode all \
    --exchange_ctpts \
    # --resume \
    # --continue_to_play \
    # --saved_epoch 8


