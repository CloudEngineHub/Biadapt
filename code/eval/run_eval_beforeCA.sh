CUDA_VISIBLE_DEVICES=2 xvfb-run -a python Dualafford_eval/eval_sampleSucc_main.py \
    --categories  Pliers \
    --primact_type opening  \
    --actor1_version model_actor_fir   \
    --actor1_path   ../logs/actor/exp-model_actor_fir-opening-Scissors-0103 \
    --actor1_eval_epoch 55 \
    --critic1_version model_critic_fir \
    --critic1_path ../logs/critic/exp-model_critic_fir-opening-Scissors-0103 \
    --critic1_eval_epoch  42 \
    --aff1_version model_aff_fir \
    --aff1_path  ../logs/affordance/exp-model_aff_fir-opening-Scissors-0103 \
    --aff1_eval_epoch 60 \
    --actor2_version model_actor_sec   \
    --actor2_path   ../logs/actor/exp-model_actor_sec-opening-Scissors-0103 \
    --actor2_eval_epoch 60 \
    --critic2_version model_critic_sec\
    --critic2_path  ../logs/critic/exp-model_critic_sec-opening-Scissors-0103 \
    --critic2_eval_epoch 60 \
    --aff2_version model_aff_sec  \
    --aff2_path  ../logs/affordance/exp-model_aff_sec-opening-Scissors-0103 \
    --aff2_eval_epoch  42 \
    --val_data_dir  ../exp/Pliers_0103_ALL \
    --val_buffer_max_num  100 \
    --coordinate_system world \
    --target_part_state closed  \
    --start_dist 0.15   \
    --final_dist 0.10   \
    --move_steps 3500   \
    --wait_steps 2000   \
    --num_processes 10 \
    --z_dim 32        \
    --repeat_num 3  \
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






