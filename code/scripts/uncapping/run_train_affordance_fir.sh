CUDA_VISIBLE_DEVICES=7 python train/train_affordance_fir.py \
    --exp_suffix 0826 \
    --category_types  Pen \
    --primact_type uncapping \
    --model_version model_aff_fir \
    --actor_version model_actor_fir  \
    --actor_path  ../logs/actor/exp-model_actor_fir-uncapping-Pen-0826 \
    --actor_eval_epoch 105 \
    --critic_version model_critic_fir   \
    --critic_path  ../logs/critic/exp-model_critic_fir-uncapping-Pen-0826 \
    --critic_eval_epoch 52 \
    --offline_data_dir  ../exp/Pen_Uncapping \
    --val_data_dir   ../exp/Pen_Uncapping \
    --train_buffer_max_num 800  \
    --val_buffer_max_num 200  \
    --feat_dim 128   \
    --batch_size 16  \
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


