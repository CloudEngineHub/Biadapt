CUDA_VISIBLE_DEVICES=2 xvfb-run -a python adapt/eval_adapt_main.py \
    --categories  Pen \
    --primact_type uncapping  \
    --trg_txt_path ../stats/uncapping_Test_list.txt \
    --ctpts_2d_file_path ../stats/ctpts_2d_uncapping_Pen.json \
    --ctpts_3d_file_path ../stats/ctpts_3d_uncapping_Pen.json \
    --out_folder adapt_eval \
    --use_CA \
    --CA_path ../logs/adaptation/exp-fewshot_tune-uncapping-Pen-0826 \
    --CA_eval_epoch 9-0 \
    --actor1_version model_actor_fir   \
    --critic1_version model_critic_fir \
    --aff1_version model_aff_fir \
    --actor2_version model_actor_sec   \
    --critic2_version model_critic_sec \
    --aff2_version model_aff_sec  \
    --val_data_dir  ../exp/Pen_Uncapping \
    --val_buffer_max_num  50 \
    --coordinate_system world \
    --target_part_state closed  \
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
    --rv1 100           \
    --rv2 100            \
    --critic_topk1 0.01 \
    --critic_topk 0.001 \
    --aff1_topk 0.5 \
    --num_pair 10 \
    --no_gui





