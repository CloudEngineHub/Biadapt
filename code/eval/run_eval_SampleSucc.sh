xvfb-run -a python Dualafford_eval/eval_sampleSucc.py  \
    --file_dir  ../exp/Laptop_1102_TRAIN/succ_files \
    --file_id  1 \
    --out_dir ../expdata/Sample_dir \
    --repeat_id 0 \
    --density 2.0 \
    --damping 10 \
    --target_part_state 'closed' \
    --start_dist 0.15 \
    --final_dist 0.10 \
    --move_steps 2000 \
    --wait_steps 2000 \
    --position_world1  'str0.13372529380794873,
        0.5471707582473755,
        0.03982134481538302' \
    --up_world1 'str0.6350324554468348,
        -0.7619941651163193,
        -0.12688054562401332'\
    --forward_world1 'str0.11501318977403716,
        -0.06915493722980315,
        0.9909538641302851'\
    --position_world2  'str-0.4231097234010539,
        -0.05689870566129679,
        0.3435467436130275'\
    --up_world2 'str0.4928186718473111,
        -0.691783570385487,
        -0.5277928082338376'\
    --forward_world2 'str0.6863048444338651,
        0.6819079376228803,
        -0.25295696296705966'\
    --save_results \
    --no_gui
