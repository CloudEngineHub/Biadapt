CUDA_VISIBLE_DEVICES=1 python train/train_affordance_sec.py \
    --exp_suffix 0826 \
    --category_types  Pen \
    --primact_type uncapping \
    --model_version model_aff_sec \
    --actor_version model_actor_sec   \
    --actor_path   ../logs/actor/exp-model_actor_sec-uncapping-Pen-0826 \
    --actor_eval_epoch  110 \
    --critic_version model_critic_sec   \
    --critic_path  ../logs/critic/exp-model_critic_sec-uncapping-Pen-0826 \
    --critic_eval_epoch 76 \
    --offline_data_dir  ../exp/Pen_Uncapping \
    --val_data_dir   ../exp/Pen_Uncapping \
    --train_buffer_max_num 8000  \
    --val_buffer_max_num 100  \
    --feat_dim 128   \
    --batch_size 32  \
    --lr 0.001      \
    --lr_decay_every 300 \
    --succ_proportion 0.5 \
    --fail_proportion 0.5 \
    --z_dim 32      \
    --topk  100       \
    --coordinate_system world \
    --mode all \
    --exchange_ctpts
    


