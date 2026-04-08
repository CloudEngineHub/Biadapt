CUDA_VISIBLE_DEVICES=1 python train/train_critic_fir.py \
    --exp_suffix 0826 \
    --category_types  Pen \
    --primact_type uncapping \
    --model_version model_critic_fir \
    --aff_version model_aff_sec   \
    --aff_path  ../logs/affordance/exp-model_aff_sec-uncapping-Pen-0826 \
    --aff_eval_epoch  32 \
    --actor_version model_actor_sec   \
    --actor_path   ../logs/actor/exp-model_actor_sec-uncapping-Pen-0826 \
    --actor_eval_epoch  110 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-model_critic_sec-uncapping-Pen-0826 \
    --critic_eval_epoch 76 \
    --offline_data_dir  ../exp/Pen_Uncapping \
    --val_data_dir   ../exp/Pen_Uncapping \
    --train_buffer_max_num 1000  \
    --val_buffer_max_num 200  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 300 \
    --topk2 1000      \
    --succ_proportion 0.5 \
    --fail_proportion 1.5 \
    --coordinate_system world \
    --loss_type L1Loss \
    --mode train \
    --exchange_ctpts













