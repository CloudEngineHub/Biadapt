CUDA_VISIBLE_DEVICES=7 python train/train_critic_fir.py \
    --exp_suffix 0910_50 \
    --category_types Pliers  \
    --primact_type opening \
    --model_version model_critic_fir \
    --aff_version model_aff_sec   \
    --aff_path  ../logs/affordance/exp-model_aff_sec-opening-Pliers-0910_50 \
    --aff_eval_epoch  200 \
     --actor_version model_actor_sec   \
    --actor_path   ../logs/actor/exp-model_actor_sec-opening-Pliers-0910_50 \
    --actor_eval_epoch  80 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-model_critic_sec-opening-Pliers-0910_50 \
    --critic_eval_epoch 80 \
    --offline_data_dir  ../exp/Pliers_Opening \
    --val_data_dir   ../exp/Pliers_Opening \
    --train_buffer_max_num 50  \
    --val_buffer_max_num 20  \
    --feat_dim 128   \
    --batch_size 8 \
    --lr 0.001      \
    --lr_decay_every 300 \
    --topk2 1000      \
    --succ_proportion 0.5 \
    --fail_proportion 1.5 \
    --coordinate_system world \
    --loss_type L1Loss \
    --mode all \
    --exchange_ctpts













