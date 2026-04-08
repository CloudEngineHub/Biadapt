CUDA_VISIBLE_DEVICES=2 python adapt/tune_actor.py \
    --exp_suffix 0910 \
    --category_types  Pliers \
    --model_version model_actor_sec \
    --primact_type opening \
    --use_CA \
    --CA_path ../logs/finetune/exp-finetune-opening-Scissors-0826 \
    --CA_eval_epoch 18-0 \
    --offline_data_dir  ../exp/sample_Scissors2Pliers_0910/Pliers_0103_ALL \
    --offline_data_dir2 ../exp/sample_Scissors2Pliers_0826/Pliers_0103_ALL \
    --offline_data_dir3 ../exp/Pliers_0103_ALL \
    --val_data_dir  ../exp/sample_Scissors2Pliers_0910/Pliers_0103_ALL \
    --val_data_dir2 ../exp/Pliers_0103_ALL \
    --train_buffer_max_num 50  \
    --val_buffer_max_num 20  \
    --feat_dim 128   \
    --batch_size 8  \
    --lr 0.001 \
    --lr_decay_every 200 \
    --lbd_kl 0.02     \
    --lbd_dir 5.0     \
    --z_dim 32   \
    --coordinate_system world \
    --mode all


