CUDA_VISIBLE_DEVICES=0 xvfb-run -a python eval/eval_sampleSucc_main.py \
    --categories  USB \
    --primact_type uncapping \
    --use_CA \
    --CA_path ../logs/finetune/exp-finetune-uncapping-Pen-0826 \
    --CA_eval_epoch 33-0 \
    --actor1_version model_actor_fir   \
    --critic1_version model_critic_fir \
    --aff1_version model_aff_fir \
    --actor2_version model_actor_sec   \
    --critic2_version model_critic_sec \
    --aff2_version model_aff_sec  \
    --val_data_dir   ../exp/USB_Uncapping \
    --val_buffer_max_num  100 \
    --coordinate_system world \
    --target_part_state "open"  \
    --start_dist 0.15   \
    --final_dist 0.10   \
    --move_steps 3500   \
    --wait_steps 2000   \
    --num_processes 10 \
    --z_dim 32        \
    --repeat_num 2  \
    --euler_threshold 10 \
    --task_threshold 30 \
    --num_ctpt1 10       \
    --num_ctpt2 10       \
    --rv1 100            \
    --rv2 100            \
    --num_pair1 10       \
    --aff_topk 0.1   \
    --critic_topk1 0.01 \
    --critic_topk 0.001 \
    --draw_aff_map \
    --no_gui






