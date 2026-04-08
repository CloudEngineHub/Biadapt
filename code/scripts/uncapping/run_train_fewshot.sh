CUDA_VISIBLE_DEVICES=0 xvfb-run -a python adapt/train_fewshot.py \
    --exp_suffix 0913 \
    --category_types Pen \
    --primact_type uncapping \
    --target_part_state 'random-right' \
    --ctpts_3d_file_path ../stats/ctpts_3d_uncapping_Pen.json \
    --use_CA \
    --CA_path ../logs/finetune/exp-finetune-uncapping-Pen-0826 \
    --CA_eval_epoch 33-0 \
    --actor1_version model_actor_fir   \
    --critic1_version model_critic_fir \
    --aff1_version model_aff_fir \
    --actor2_version model_actor_sec   \
    --critic2_version model_critic_sec\
    --aff2_version model_aff_sec  \
    --offline_data_dir ../exp/Pen_Uncapping \
    --offline_data_dir2 ../exp/sample_USB2Pen_0826/Pen_Uncapping \
    --val_data_dir ../exp/sample_USB2Pen_0826/Pen_Uncapping \
    --target_part_state closed  \
    --out_folder finetune_adaptation \
    --epochs 10 \
    --sample_max_num 30 \
    --num_processes 10 \
    --train_buffer_max_num 300  \
    --val_buffer_max_num 300  \
    --succ_proportion 0.8 \
    --fail_proportion 0.4 \
    --feat_dim 128   \
    --batch_size 32 \
    --lr 0.001      \
    --lr_decay_every 20 \
    --critic1_lr 0.0001 \
    --critic1_lr_decay_every 50 \
    --aff_lr 0.0001     \
    --aff_lr_decay_every 50 \
    --loss1_weight 1.0  \
    --loss2_weight 0.0 \
    --num_ctpt1 10  \
    --num_ctpt2 10  \
    --num_pair1 10  \
    --rv1 100        \
    --rv2 100        \
    --z_dim 32      \
    --start_dist 0.45   \
    --final_dist 0.10   \
    --move_steps 3500   \
    --wait_steps 2000   \
    --coordinate_system world \
    --euler_threshold 3 \
    --task_threshold 30 \
    --aff_topk 0.5   \
    --critic_topk 0.001 \
    --critic_topk1 0.01 \
    --loss_type L1Loss
    # --resume \
    # --saved_epoch 1-0 \
    # --continue_to_play
    # # --num_interaction_data_offline 5 



