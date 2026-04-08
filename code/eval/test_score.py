import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
import utils
import json
from argparse import ArgumentParser
import torch
import torch.utils.data
import torch.nn.functional as F
import time
import random
from subprocess import call
from data import SAPIENVisionDataset
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import math
import datetime
import h5py
from tensorboardX import SummaryWriter

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp_suffix', type=str, help='exp suffix')
parser.add_argument('--category_types',
                    type=str,
                    help='list all category_types [Default: None, meaning all 10 category_types]',
                    default=None)
parser.add_argument('--cat2freq', type=str, default=None)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_folder', type=str, default='xxx')
parser.add_argument('--repeat_num', type=int, default=1)
parser.add_argument('--seed',
                    type=int,
                    default=-1,
                    help='random seed (for reproducibility) [specify -1 means to generate a random one]')

parser.add_argument('--val_data_dir', type=str, help='data directory')
parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
parser.add_argument('--val_buffer_max_num', type=int, default=20000)
parser.add_argument('--val_cat2freq', type=str, default=None)
# network settings
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--num_point_per_shape', type=int, default=8192)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--task_feat_dim', type=int, default=32)
parser.add_argument('--cp_feat_dim', type=int, default=32)
parser.add_argument('--dir_feat_dim', type=int, default=32)
parser.add_argument('--rv_cnt', type=int, default=100)
parser.add_argument('--rvs_ctpt', type=int, default=10)
parser.add_argument('--rvs', type=int, default=100)
parser.add_argument('--num_ctpt1', type=int, default=10)
parser.add_argument('--num_ctpt2', type=int, default=10)
parser.add_argument('--num_pair1', type=int, default=10)
parser.add_argument('--rv1', type=int, default=10)
parser.add_argument('--rv2', type=int, default=10)
parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--topk2', type=int, default=1000)
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--no_true_false_equal',
                    action='store_true',
                    default=False,
                    help='if make the true/false data loaded equally [default: False]')
parser.add_argument('--coordinate_system', type=str, default='world')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--succ_proportion', type=float, default=0.50)
parser.add_argument('--fail_proportion', type=float, default=0.75)
parser.add_argument('--exchange_ctpts', action='store_true', default=False)
parser.add_argument('--task_threshold', type=int, default=15)
parser.add_argument('--euler_threshold', type=int, default=5)

parser.add_argument('--critic2_version', type=str, default=None, help='model def file')
parser.add_argument('--critic2_path', type=str)
parser.add_argument('--critic2_eval_epoch', type=str)
parser.add_argument('--critic_score_threshold', type=float, default=0.5)

parser.add_argument('--CA_path', type=str)
parser.add_argument('--CA_eval_epoch', type=str)
parser.add_argument('--use_CA', action='store_true', default=False)

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=0)
# logging
parser.add_argument('--no_tb_log', action='store_true', default=False)
parser.add_argument('--no_console_log', action='store_true', default=False)
parser.add_argument('--console_log_interval',
                    type=int,
                    default=2,
                    help='number of optimization steps beween console log prints')

args = parser.parse_args()
ctx = torch.multiprocessing.get_context("spawn")


def test(conf, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['part_pc', 'ctpt1', 'ctpt2', 'dir1', 'dir2', 'success', 'shape_id', 'trial_id']
    primact_type = args.primact_type
    task_input_dim = 1
    device = conf.device

    critic2_def = utils.get_model_module(args.critic2_version)
    critic2 = critic2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    # if not args.use_CA:
    #     critic2.load_state_dict(
    #         torch.load(os.path.join(args.critic2_path, 'ckpts', '%s-network.pth' % args.critic2_eval_epoch)))
    # else:
    #     critic2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic2.pth' % args.CA_eval_epoch)))

    # critic2.to(device).eval()
    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        # train_writer = SummaryWriter(os.path.join(conf.tb_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.tb_dir, 'val'))

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.val_buffer_max_num, \
                                      img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal,
                                      succ_proportion=conf.succ_proportion, fail_proportion=conf.fail_proportion,
                                      coordinate_system=conf.coordinate_system, exchange_ctpts=conf.exchange_ctpts,
                                      cat2freq=conf.val_cat2freq)

    val_dataset.load_data(val_data_list)
    print('aaaaaaaaaadď11--------------------------------------')
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=conf.batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 drop_last=True,
                                                 collate_fn=utils.collate_feats,
                                                 worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)
    print('val_num_batch: %d' % (val_num_batch))
    # start test
    start_time = time.time()
    start_epoch = conf.start_epoch
    end_epoch = conf.end_epoch
    #load model
    last_val_console_log_step = None
    for model_epoch in range(start_epoch, end_epoch + 1):

        eval_epoch = model_epoch  #TODO

        if not args.use_CA:

            data_to_restore = os.path.join(args.critic2_path, 'ckpts', '%s-network.pth' % str(eval_epoch))

        else:
            eval_epoch = str(model_epoch) + '-0'
            data_to_restore = os.path.join(args.CA_path, 'ckpts', '%s-critic2.pth' % str(eval_epoch))

        critic2.load_state_dict(torch.load(data_to_restore))
        critic2.to(device).eval()
        print('loading model from: ', data_to_restore)

        total_pred_and_gt = 0
        total_pred_sum = 1
        total_gt_sum = 1
        correct_sum = 0
        all_sum = 0

        val_batches = enumerate(val_dataloader, 0)
        ### test for every batch
        for val_batch_ind, batch in val_batches:
            val_fraction_done = (val_batch_ind + 1) / val_num_batch
            val_step = (model_epoch + val_fraction_done) * val_num_batch - 1

            log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                    val_step - last_val_console_log_step >= conf.console_log_interval)
            if log_console:
                last_val_console_log_step = val_step
            with torch.no_grad():
                pred_and_gt, pred_sum, gt_sum, correct = forward(batch=batch,
                                                                 data_features=data_features,
                                                                 network=critic2,
                                                                 conf=conf,
                                                                 is_val=True,
                                                                 batch_ind=val_batch_ind)
            total_pred_and_gt += pred_and_gt
            total_pred_sum += pred_sum
            total_gt_sum += gt_sum
            correct_sum += correct
            all_sum += conf.batch_size

        precise = total_pred_and_gt / total_pred_sum
        recall = total_pred_and_gt / total_gt_sum
        F1 = 2 * (precise * recall) / (precise + recall)

        precise = round(precise, 3)
        recall = round(recall, 3)
        F1 = round(F1, 3)

        print('score from: ', conf.out_dir + f'-{model_epoch}', str(conf.category_types))
        print('Score:  ', precise, '   ', recall, '   ', F1)

        with open(os.path.join(conf.out_dir, 'F1_score.txt'), 'a') as fout:
            if model_epoch == conf.start_epoch:
                fout.write(f'{conf.val_data_dir} {str(conf.category_types)} Score: \n')
            fout.write(f'{model_epoch}   {precise}   {recall}   {F1}\n')


def forward(batch,
            data_features,
            network,
            conf,
            is_val=False,
            step=None,
            epoch=None,
            batch_ind=0,
            num_batch=1,
            start_time=0,
            log_console=False,
            log_tb=False,
            tb_writer=None,
            lr=None):
    batch_size = conf.batch_size
    device = conf.device
    pcs = torch.cat(batch[data_features.index('part_pc')], dim=0).float().to(device)
    # pcs = torch.tensor(batch[data_features.index('part_pc')]).float().view(batch_size, -1, 3).to(device)
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    pcs = pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

    ctpt1 = torch.tensor(np.array(batch[data_features.index('ctpt1')])).float().view(batch_size, -1).to(device)
    ctpt2 = torch.tensor(np.array(batch[data_features.index('ctpt2')])).float().view(batch_size, -1).to(device)
    dir1 = torch.tensor(np.array(batch[data_features.index('dir1')])).float().view(batch_size, -1).to(device)
    dir2 = torch.tensor(np.array(batch[data_features.index('dir2')])).float().view(batch_size, -1).to(device)

    gt_labels = torch.tensor(np.array(batch[data_features.index('success')])).float().view(batch_size, -1).to(device)

    gt_result = torch.Tensor(batch[data_features.index('success')]).long().numpy()  # B

    # input to the pipeline
    #    x if conf.model_version == 'model_critic_sec':
    pred_score = network.forward(pcs, ctpt1, ctpt2, dir1, dir2)  # after sigmoid
    loss = network.get_ce_loss_total(pred_score, gt_labels)
    total_loss = loss.mean()

    # if is_val:
    pred = pred_score.detach().cpu().numpy() > conf.critic_score_threshold
    Fscore, precision, recall, accu = utils.cal_Fscore(np.array(pred, dtype=np.int32), gt_labels.detach().cpu().numpy())

    # compute correctness
    pred_and_gt = np.logical_and(pred, gt_result)

    correct = (pred == gt_result)
    # display information
    data_split = 'val' if is_val else 'train'
    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                           f'''{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('critic_loss', total_loss.item(), step)
            # tb_writer.add_scalar('critic_lr', lr, step)
        if is_val and log_tb and tb_writer is not None:
            tb_writer.add_scalar('Fscore', Fscore, step)
            tb_writer.add_scalar('precision', precision, step)
            tb_writer.add_scalar('recall', recall, step)
            tb_writer.add_scalar('accu', accu, step)

    return pred_and_gt.sum(), pred.sum(), gt_result.sum(), correct.sum()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    ### get parameters
    if args.use_CA:
        out_dir = os.path.join(args.CA_path, args.out_folder)
    else:
        out_dir = os.path.join(args.critic2_path, args.out_folder)
    print('out_dir: ', out_dir)
    if os.path.exists(out_dir):
        response = input('A testing score run named "%s" already exists, overwrite? (y/n) ' % out_dir)
        if response != 'y':
            exit(1)
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    dir_list = [
        args.val_data_dir, args.val_data_dir2, args.val_data_dir3, args.val_data_dir4, args.val_data_dir5,
        args.val_data_dir6
    ]
    conf = parser.parse_args()
    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    conf.out_dir = out_dir
    conf.tb_dir = os.path.join(conf.out_dir, 'tb')
    os.makedirs(conf.tb_dir, exist_ok=True)
    flog = open(os.path.join(out_dir, 'test_log.txt'), 'w')
    conf.flog = flog
    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))

    val_data_list = []
    val_data_dir_list = [
        conf.val_data_dir, conf.val_data_dir2, conf.val_data_dir3, conf.val_data_dir4, conf.val_data_dir5,
        conf.val_data_dir6
    ]
    for data_dir in val_data_dir_list:
        val_data_list.extend(utils.append_data_list(data_dir, primact_type=conf.primact_type))
    print('0-----------------------------------------------')
    utils.printout(flog, 'len(val_data_list): %d' % len(val_data_list))
    print('val_data_list: ')
    for idx in range(len(val_data_list)):
        print(val_data_list[idx])
    ###start valuating
    test(conf, val_data_list)
    flog.close()
