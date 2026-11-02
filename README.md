# Bi-Adapt: Few-shot Bimanual Adaptation for Novel Categories of 3D Objects via Semantic Correspondence

![Bi-Adapt overview](images/overview.pdf)

[Open the overview PDF](images/overview.pdf)

We present Bi-Adapt, a novel framework designed for efficient learning of generalizable bimanual manipulation. It first learns point-level action on the supporting set for different bimanual tasks, then it predicts actions on novel categories based on the foundation-model-guided affordance, enabling cross-category generalization after few-shot adaptation.

## About the paper

---

Bi-Adapt is accepted by ICRA 2026.

<div class="is-size-5 publication-authors">
  <span class="author-block">
    <a href="https://isabella4444x.github.io">Jinxian Zhou</a><sup>*</sup>,
    <a href="https://warshallrho.github.io/">Ruihai Wu</a><sup>*</sup>,
    <a href="https://lew1sin.github.io/">Yiwei Liu</a>,
    <a href="https://houyiwen.github.io/">Yiwen Hou</a>,
    <a href="https://zhouxunzhe.github.io/">Xunzhe Zhou</a>,
    <a href="https://tradewindycc.github.io/">Checheng Yu</a>,
    <a href="https://zlicheng.com">Licheng Zhong</a>,
    <a href="https://linsats.github.io/">Lin Shao</a><sup>†</sup>
  </span>
</div>

<div class="is-size-6 publication-authors">
  <span class="author-block">
    <sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.
  </span>
</div>

Project page: https://biadapt-project.github.io/

ArXiv: http://arxiv.org/abs/2602.08425

## Before start

---

To train the models, please first go to the data folder and download the pre-processed SAPIEN dataset for Bi-Adapt. To test over the pretrained models, please go to the logs folder and download the pretrained checkpoints.

You can click [here](https://isabella4444x.github.io/) to download all resources.

The public repository does not include large datasets, generated experiment results, or pretrained checkpoints. The code assumes a workspace layout like:

```text
Biadapt/
├── code/
├── stats/
├── exp/
└── logs/
```

In the current scripts, exp/ is used for generated data, correspondence outputs, and transferred contact points; logs/ is used for checkpoints and training logs.

The split files in stats/ use the original task naming from the research codebase, including filenames such as openning_Train_list.txt.

## Dependencies

---

This code has been tested on Ubuntu 20.04.6 with CUDA 11.7, Python 3.8, and PyTorch 2.4.0.

First, install SAPIEN:

```bash
pip install sapien
```

Then, if you want to run the 3D experiments, this code depends on PointNet++.
The current repository already includes the relevant PointNet++ sources under code/models/pointnet2 and code/models/pointnet2_ops. After activating the environment, please make sure the code directory is in your Python path:

```bash
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
pip install -r requirements.txt
pip install -e .
```

Finally, run the following to install the other Python packages:

```bash
# make sure you are at the repository root directory
pip install -r requirements.txt
```

For visualization, please install Blender and put the executable in your environment path. The prediction results can also be visualized using tools such as MeshLab.

## Stage1: Training On Supporting Set

---

### Generate Offline Training Data

Before training, collect interaction trials in simulation. The data collection entry points are under `code/collect/`.

The released example script is:

```bash
cd code
bash collect/run_collect.sh
```

`run_collect.sh` is a command template for launching `collect/collect_data_main.py` with `xvfb-run` for headless SAPIEN rendering. Some template scripts may still contain legacy directory names from the research codebase, so first make sure paths such as `Collect/` are changed to the current lowercase `collect/` directory. Edit the script for your task and category before running. The most important options are:

- `--category`: object category to sample, for example `Pen`, `Scissors`, or `Pliers`.
- `--primact_type`: manipulation task, for example `capping` or `opening`.
- `--out_dir`: output directory under `exp/`.
- `--mode`: split to sample from, usually `train`, `val`, or `all`.
- `--target_part_state`, `--start_dist`, `--final_dist`, `--move_steps`, `--wait_steps`: simulation/task settings.
- `--num_processes`: number of parallel CPU workers.
- `--save_interval`: interval for saving trial files.
- `--no_gui`: run without GUI.

For example:

```bash
cd code
xvfb-run -a python collect/collect_data_main.py \
    --category Pen \
    --primact_type capping \
    --out_dir ../exp/Pen_capping_train \
    --mode train \
    --target_part_state open \
    --start_dist 0.15 \
    --final_dist 0.10 \
    --move_steps 2000 \
    --wait_steps 2000 \
    --num_processes 10 \
    --start_epoch 0 \
    --save_interval 10 \
    --not_check_dual \
    --no_gui
```

The generated folder contains successful and failed interaction files, images, and gifs, for example `succ_files/`, `fail_files/`, `succ_img/`, and `succ_gif/`. These folders are later passed to the training scripts through `--offline_data_dir` and `--val_data_dir`.

### Training Pipeline

The Stage 1 training code is in `code/train/`. Bi-Adapt uses two gripper stages, each with an actor model and an affordance model. In practice, the affordance model is trained with a pretrained actor and critic checkpoint, so a minimal training order is:

1. Train the second gripper actor/critic.
2. Train the second gripper affordance model with the trained second actor/critic.
3. Train the first gripper actor/critic.
4. Train the first gripper affordance model with the trained first actor/critic.

Use the generated data folders from the previous step as `--offline_data_dir` and `--val_data_dir`. The model names used by the released code are:

```text
model_actor_sec, model_critic_sec, model_aff_sec
model_actor_fir, model_critic_fir, model_aff_fir
```

Example commands:

```bash
cd code

python train/train_critic_sec.py \
    --exp_suffix demo \
    --model_version model_critic_sec \
    --primact_type opening \
    --category_types Scissors \
    --offline_data_dir ../exp/Scissors_opening_train \
    --val_data_dir ../exp/Scissors_opening_val \
    --log_dir ../logs/critic \
    --coordinate_system world \
    --z_dim 32 \
    --batch_size 32

python train/train_actor_sec.py \
    --exp_suffix demo \
    --model_version model_actor_sec \
    --primact_type opening \
    --category_types Scissors \
    --offline_data_dir ../exp/Scissors_opening_train \
    --val_data_dir ../exp/Scissors_opening_val \
    --log_dir ../logs/actor \
    --coordinate_system world \
    --z_dim 32 \
    --batch_size 32

python train/train_affordance_sec.py \
    --exp_suffix demo \
    --model_version model_aff_sec \
    --primact_type opening \
    --category_types Scissors \
    --offline_data_dir ../exp/Scissors_opening_train \
    --val_data_dir ../exp/Scissors_opening_val \
    --actor_version model_actor_sec \
    --actor_path ../logs/actor/exp-model_actor_sec-opening-Scissors-demo \
    --actor_eval_epoch 60 \
    --critic_version model_critic_sec \
    --critic_path ../logs/critic/exp-model_critic_sec-opening-Scissors-demo \
    --critic_eval_epoch 60 \
    --log_dir ../logs/affordance \
    --coordinate_system world \
    --z_dim 32 \
    --batch_size 32
```

Run the corresponding first-gripper scripts by replacing `sec` with `fir`:

```bash
python train/train_actor_fir.py ...
python train/train_affordance_fir.py ...
```

The critic scripts are also provided in the same directory:

```bash
python train/train_critic_fir.py ...
```

Training outputs are written to `logs/actor/`, `logs/critic/`, and `logs/affordance/`. Checkpoints are saved under each experiment's `ckpts/` folder.

### Optional: Collaborative Adaptation (CA)

After training the first and second affordance, actor, and critic models, you can optionally run collaborative adaptation on the supporting set. This jointly fine-tunes the two-stage policy modules and writes CA checkpoints to `logs/finetune/`. These checkpoints are later enabled in Stage 3 and evaluation with `--use_CA`, `--CA_path`, and `--CA_eval_epoch`.

```bash
cd code
CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train/train_collaborative_adaptation.py \
    --exp_suffix demo \
    --model_version finetune \
    --primact_type opening \
    --category_types Scissors \
    --offline_data_dir ../exp/Scissors_opening_train \
    --val_data_dir ../exp/Scissors_opening_val \
    --out_folder CA_rollouts \
    --aff1_version model_aff_fir \
    --aff1_path ../logs/affordance/exp-model_aff_fir-opening-Scissors-demo \
    --aff1_eval_epoch 60 \
    --actor1_version model_actor_fir \
    --actor1_path ../logs/actor/exp-model_actor_fir-opening-Scissors-demo \
    --actor1_eval_epoch 60 \
    --critic1_version model_critic_fir \
    --critic1_path ../logs/critic/exp-model_critic_fir-opening-Scissors-demo \
    --critic1_eval_epoch 60 \
    --aff2_version model_aff_sec \
    --aff2_path ../logs/affordance/exp-model_aff_sec-opening-Scissors-demo \
    --aff2_eval_epoch 60 \
    --actor2_version model_actor_sec \
    --actor2_path ../logs/actor/exp-model_actor_sec-opening-Scissors-demo \
    --actor2_eval_epoch 60 \
    --critic2_version model_critic_sec \
    --critic2_path ../logs/critic/exp-model_critic_sec-opening-Scissors-demo \
    --critic2_eval_epoch 60 \
    --log_dir ../logs/finetune \
    --coordinate_system world \
    --z_dim 32 \
    --batch_size 32 \
    --update_batch_size 32 \
    --aff_topk 0.1 \
    --critic_topk1 0.01 \
    --critic_topk 0.01 \
    --num_ctpt1 10 \
    --num_ctpt2 10 \
    --rv1 100 \
    --rv2 100 \
    --num_pair1 10
```

The saved checkpoint names look like `18-0-affordance1.pth`, `18-0-actor1.pth`, `18-0-critic1.pth`, `18-0-affordance2.pth`, `18-0-actor2.pth`, and `18-0-critic2.pth`. When using CA later, pass the shared prefix without the module suffix, for example:

```bash
--use_CA \
--CA_path ../logs/finetune/exp-finetune-opening-Scissors-demo \
--CA_eval_epoch 18-0
```

## Stage2: Affordance Mapping

---

Stage 2 transfers contact-point affordance from a supporting category to a novel target category using semantic correspondence. The code is in `code/correspondence/`.

For interactive point checking and visualization, open:

```text
code/correspondence/dift_interactive_heatmap.ipynb
```

Use the notebook to inspect DIFT heatmaps, click or verify source/target correspondences, and debug whether the semantic matching is reasonable for the selected object pair.

For batch mapping, use:

```bash
cd code
bash correspondence/run_mapctpts.sh
```

`run_mapctpts.sh` is also a template; if it contains `Correspondence/`, change it to `correspondence/`. You can also run `mapctpts.py` directly:

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python correspondence/mapctpts.py \
    --method dift \
    --dataset clip_b32 \
    --base_dir ../exp/Retrieve_Pliers_0103 \
    --out_dir ../exp/Cor_res/Pliers_demo \
    --top_k 10 \
    --top_k_type max \
    --transpose_types 8 \
    --visualize
```

Important options:

- `--method`: correspondence backend. Supported values include `dift`, `ldm_sc`, `sd_dino`, and `dino_vit`.
- `--base_dir`: retrieved support examples and target images.
- `--out_dir`: correspondence result directory.
- `--top_k`: number of retrieved support examples used for mapping.
- `--top_k_type`: `max` selects the strongest match; `avg` averages top-k results.
- `--transpose_types`: use `1` for no augmentation, `4` for rotations, and `8` for rotations plus flips.
- `--visualize`: save correspondence visualizations and diagnostic plots.

The script writes results to:

```text
<out_dir>/results/<method>/<setting_name>/
```

This folder contains `results.pkl`, `total_dists.txt`, optional visualizations, and mapped target contact points used in the adaptation stage.

## Stage3: Few-shot Adaptation

---

Stage 3 is implemented in `code/adapt/`. It takes the Stage 2 mapped contact points and pretrained Stage 1 models, then samples and evaluates candidate bimanual actions on the novel category.

The main script is:

```bash
cd code
bash adapt/run_transfer_affordance_CA.sh
```

The example script calls `adapt/transfer_affordance.py`. Edit the paths and category/task settings before running:

```bash
cd code
CUDA_VISIBLE_DEVICES=0 xvfb-run -a python adapt/transfer_affordance.py \
    --base_dir ../exp/Cor_res/Pliers_demo/results/dift/clip_b32_top10_max_transpose \
    --out_dir ../exp/sample_Scissors2Pliers_demo \
    --categories Pliers \
    --primact_type opening \
    --target_part_state random-right \
    --setting_data_path ../stats/shape_setting.json \
    --trg_txt_path ../stats/openning_Test_list.txt \
    --ctpts_2d_file_path ../stats/ctpts_2d_opening_Pliers.json \
    --ctpts_3d_file_path ../stats/ctpts_3d_opening_Pliers.json \
    --ctpt_type txt \
    --actor1_version model_actor_fir \
    --critic1_version model_critic_fir \
    --aff1_version model_aff_fir \
    --actor2_version model_actor_sec \
    --critic2_version model_critic_sec \
    --aff2_version model_aff_sec \
    --val_data_dir ../exp/Pliers_0103_ALL \
    --val_buffer_max_num 50 \
    --coordinate_system world \
    --z_dim 32 \
    --repeat_num 2 \
    --num_processes 10 \
    --draw_aff_map \
    --no_gui
```

If you have a collaborative-adaptation checkpoint, add:

```bash
--use_CA \
--CA_path ../logs/finetune/exp-finetune-opening-Scissors-demo \
--CA_eval_epoch 18-0
```

The adaptation output directory stores sampled target-category trials, transferred contact points, rendered affordance maps, and simulation results.

`code/adapt/tune_critic_sec.py` and `code/adapt/run_tune_critic_sec.sh` are provided for tuning the second critic during few-shot adaptation. Use them when you want to further adapt the critic on sampled target-category trials.

## Evaluation

---

Evaluation scripts are under `code/eval/` and `code/adapt/`.

To evaluate pretrained Stage 1 models before few-shot adaptation, use `eval/eval_sampleSucc_main.py`. The provided `eval/run_eval_beforeCA.sh` is a template; if it contains legacy paths such as `Dualafford_eval/`, change them to `eval/` first.

```bash
cd code
CUDA_VISIBLE_DEVICES=0 xvfb-run -a python eval/eval_sampleSucc_main.py \
    --categories Pliers \
    --primact_type opening \
    --actor1_version model_actor_fir \
    --actor1_path ../logs/actor/exp-model_actor_fir-opening-Scissors-demo \
    --actor1_eval_epoch 60 \
    --critic1_version model_critic_fir \
    --critic1_path ../logs/critic/exp-model_critic_fir-opening-Scissors-demo \
    --critic1_eval_epoch 60 \
    --aff1_version model_aff_fir \
    --aff1_path ../logs/affordance/exp-model_aff_fir-opening-Scissors-demo \
    --aff1_eval_epoch 60 \
    --actor2_version model_actor_sec \
    --actor2_path ../logs/actor/exp-model_actor_sec-opening-Scissors-demo \
    --actor2_eval_epoch 60 \
    --critic2_version model_critic_sec \
    --critic2_path ../logs/critic/exp-model_critic_sec-opening-Scissors-demo \
    --critic2_eval_epoch 60 \
    --aff2_version model_aff_sec \
    --aff2_path ../logs/affordance/exp-model_aff_sec-opening-Scissors-demo \
    --aff2_eval_epoch 60 \
    --val_data_dir ../exp/Pliers_0103_ALL \
    --val_buffer_max_num 100 \
    --coordinate_system world \
    --target_part_state closed \
    --z_dim 32 \
    --repeat_num 3 \
    --num_processes 10 \
    --draw_aff_map \
    --no_gui
```

This loads the first/second actor, critic, and affordance checkpoints, proposes bimanual actions in simulation, and saves rollout results and optional affordance-map visualizations.

To evaluate after Stage 3 adaptation, use:

```bash
cd code
python adapt/eval_adapt_main.py \
    --categories Pliers \
    --primact_type opening \
    --out_folder eval_after_adapt \
    --trg_txt_path ../stats/openning_Test_list.txt \
    --ctpts_2d_file_path ../stats/ctpts_2d_opening_Pliers.json \
    --ctpts_3d_file_path ../stats/ctpts_3d_opening_Pliers.json \
    --use_CA \
    --CA_path ../logs/finetune/exp-finetune-opening-Scissors-demo \
    --CA_eval_epoch 18-0 \
    --val_data_dir ../exp/Pliers_0103_ALL \
    --val_buffer_max_num 100 \
    --coordinate_system world \
    --z_dim 32 \
    --repeat_num 3 \
    --num_processes 10 \
    --draw_aff_map \
    --no_gui
```

For critic-score evaluation over a checkpoint range, edit and run:

```bash
cd code
bash eval/run_test_score.sh
```

The most useful outputs to inspect are the success statistics printed by the scripts, saved rollout files, gifs, and affordance-map images in the selected output folder.

## Citation

---

If you find this repository useful, please cite:

```bibtex
@INPROCEEDINGS{11695815,
  author={Zhou, Jinxian and Wu, Ruihai and Liu, Yiwei and Hou, Yiwen and Zhou, Xunzhe and Yu, Checheng and Zhong, Licheng and Shao, Lin},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Bi-Adapt: Few-shot Bimanual Adaptation for Novel Categories of 3D Objects via Semantic Correspondence},
  year={2026},
  volume={},
  number={},
  pages={11683-11690},
  keywords={Modeling;Training;Contacts;Printing;Foundation models;Grippers;Learning (artificial intelligence);Arm;Aluminum;Modules (abstract algebra)},
  doi={10.1109/ICRA57385.2026.11695815}
}
```

## Questions

---

If you encounter issues when using the released code, please open a GitHub issue with the command you ran, the edited data and checkpoint paths, and the full error log.

## License

---

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
