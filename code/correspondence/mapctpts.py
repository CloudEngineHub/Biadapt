import os
import re
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import logging
import argparse
import shutil
import json
import matplotlib

matplotlib.use('Agg')


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def get_cor_cfg(method):
    cor_cfg = {}
    if method == 'dift':
        cor_cfg['img_size'] = 768
        cor_cfg['ensemble_size'] = 8
    elif method == 'ldm_sc':
        cor_cfg['img_size'] = 512
    elif method == 'sd_dino':
        cor_cfg['model_type'] = 'dinov2_vitb14'
    elif method == 'dino_vit':
        cor_cfg['img_size'] = 256
        cor_cfg['model_type'] = 'dino_vits8'
        cor_cfg['stride'] = 4
    return cor_cfg


def get_cor_pairs(method,
                  model,
                  src_image,
                  trg_image,
                  src_points,
                  src_prompt,
                  trg_prompt,
                  cfg,
                  transpose_img_func=lambda x: x,
                  transpose_pts_func=lambda x, y: (x, y),
                  device='cuda'):
    if method == 'dift':
        from sc_models.dift.get_cor import get_cor_pairs
        return get_cor_pairs(model,
                             src_image,
                             trg_image,
                             src_points,
                             src_prompt,
                             trg_prompt,
                             cfg['img_size'],
                             cfg['ensemble_size'],
                             return_cos_maps=cfg['visualize'],
                             transpose_img_func=transpose_img_func,
                             transpose_pts_func=transpose_pts_func)
    elif method == 'ldm_sc':  # ldm_sc don't get transpose_img_func and transpose_pts_func cause it will be too slow.
        from sc_models.ldm_sc.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], device), None
    elif method == 'sd_dino':
        from sc_models.sd_dino.get_cor import get_cor_pairs
        model, aug, extractor = model
        return get_cor_pairs(model,
                             aug,
                             extractor,
                             src_image,
                             trg_image,
                             src_points,
                             src_prompt,
                             trg_prompt,
                             transpose_img_func=transpose_img_func,
                             transpose_pts_func=transpose_pts_func,
                             device=device)
    elif method == 'dino_vit':
        from sc_models.dino_vit.get_cor import get_cor_pairs
        return get_cor_pairs(model,
                             src_image,
                             trg_image,
                             src_points,
                             cfg['img_size'],
                             transpose_img_func=transpose_img_func,
                             transpose_pts_func=transpose_pts_func,
                             device=device)
    else:
        raise NotImplementedError


def get_dual_cor_pairs(method,
                       model,
                       src_image,
                       trg_image,
                       src_points1,
                       src_points2,
                       src_prompt,
                       trg_prompt,
                       cfg,
                       transpose_img_func=lambda x: x,
                       transpose_pts_func=lambda x, y: (x, y),
                       device='cuda'):
    if method == 'dift':
        from sc_models.dift.get_cor import get_dual_cor_pairs
        return get_dual_cor_pairs(model,
                                  src_image,
                                  trg_image,
                                  src_points1,
                                  src_points2,
                                  src_prompt,
                                  trg_prompt,
                                  cfg['img_size'],
                                  cfg['ensemble_size'],
                                  return_cos_maps=cfg['visualize'],
                                  transpose_img_func=transpose_img_func,
                                  transpose_pts_func=transpose_pts_func)
    # elif method == 'ldm_sc':  # ldm_sc don't get transpose_img_func and transpose_pts_func cause it will be too slow.
    #     from sc_models.ldm_sc.get_cor import get_cor_pairs
    #     return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], device), None
    elif method == 'sd_dino':
        from sc_models.sd_dino.get_cor import get_dual_cor_pairs
        model, aug, extractor = model
        return get_dual_cor_pairs(model,
                                  aug,
                                  extractor,
                                  src_image,
                                  trg_image,
                                  src_points1,
                                  src_points2,
                                  src_prompt,
                                  trg_prompt,
                                  transpose_img_func=transpose_img_func,
                                  transpose_pts_func=transpose_pts_func,
                                  device=device)
    elif method == 'dino_vit':
        from sc_models.dino_vit.get_cor import get_dual_cor_pairs
        return get_dual_cor_pairs(model,
                                  src_image,
                                  trg_image,
                                  src_points1,
                                  src_points2,
                                  cfg['img_size'],
                                  transpose_img_func=transpose_img_func,
                                  transpose_pts_func=transpose_pts_func,
                                  device=device)
    else:
        raise NotImplementedError


def get_model(method, cor_cfg, device='cuda'):
    if method == 'dift':
        from sc_models.dift.dift_sd import SDFeaturizer
        return SDFeaturizer(device)
    elif method == 'ldm_sc':
        from sc_models.ldm_sc.optimize import load_ldm
        return load_ldm(device, 'CompVis/stable-diffusion-v1-4')
    elif method == 'sd_dino':
        from sc_models.sd_dino.extractor_sd import load_model
        from sc_models.sd_dino.extractor_dino import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = 14 if 'v2' in model_type else 8
        extractor = ViTExtractor(model_type, stride, device=device)
        model, aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100, block_indices=(2, 5, 8, 11))
        return model, aug, extractor
    elif method == 'dino_vit':
        from sc_models.dino_vit.extractor import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = cor_cfg['stride']
        return ViTExtractor(model_type, stride, device=device)


def plot_img_pairs(imglist,
                   src_points_list,
                   trg_points_list,
                   trg_mask,
                   top_k=1,
                   cos_map_list=None,
                   save_name='corr.png',
                   fig_size=5,
                   alpha=0.45,
                   scatter_size=30):
    num_imgs = top_k + 1
    src_images = len(imglist) - 1
    fig, axes = plt.subplots(src_images, num_imgs + 1, figsize=(fig_size * (num_imgs + 1), fig_size * src_images))
    plt.tight_layout()

    for i in range(src_images):
        ax = axes[i] if src_images > 1 else axes
        ax[0].imshow(imglist[i])
        ax[0].axis('off')
        ax[0].set_title('source')
        for x, y in src_points_list[i]:
            x, y = int(np.round(x)), int(np.round(y))
            ax[0].scatter(x, y, s=scatter_size)

        for j in range(1, num_imgs):
            ax[j].imshow(imglist[-1])
            if cos_map_list[0] is not None:
                print("i,j: ", i, j)
                heatmap = cos_map_list[i][j - 1][0]
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                ax[j].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
            ax[j].axis('off')
            ax[j].scatter(trg_points_list[i][j - 1][0], trg_points_list[i][j - 1][1], c='C%d' % (j - 1), s=scatter_size)
            ax[j].set_title('target')

        ax[-1].imshow(trg_mask, cmap='gray')
        ax[-1].axis('off')
        ax[-1].set_title('target mask')
        trg_point = np.mean(trg_points_list[i], axis=0)
        ax[-1].scatter(trg_point[0], trg_point[1], c='C%d' % (j - 1), s=scatter_size)
    plt.plot()
    plt.savefig(save_name)
    plt.close()


def plot_dual_img_pairs(imglist,
                        src_point1_list,
                        trg_point1_list,
                        src_point2_list,
                        trg_point2_list,
                        trg_mask,
                        top_k=1,
                        cos_map1_list=None,
                        cos_map2_list=None,
                        save_name='corr.png',
                        fig_size=5,
                        alpha=0.45,
                        scatter_size=30):

    num_imgs = top_k * 2 + 1
    src_images = len(imglist) - 1
    fig, axes = plt.subplots(src_images, num_imgs + 1, figsize=(fig_size * (num_imgs + 1), fig_size * src_images))
    plt.tight_layout()
    reds = cm.get_cmap('Reds')
    blues = cm.get_cmap('Blues')
    for i in range(src_images):
        ax = axes[i] if src_images > 1 else axes
        ax[0].imshow(imglist[i])
        ax[0].axis('off')
        ax[0].set_title('source')

        for x, y in src_point1_list[i]:
            x, y = int(np.round(x)), int(np.round(y))
            ax[0].scatter(x, y, c='red', s=scatter_size, label='Point 1')

        for x, y in src_point2_list[i]:
            x, y = int(np.round(x)), int(np.round(y))
            ax[0].scatter(x, y, c='blue', s=scatter_size, label='Point 2')

        for k in range(1, top_k + 1):

            j = k * 2 - 1
            ax[j].imshow(imglist[-1])
            if cos_map1_list[0] is not None:
                print("i,j,k: ", i, j, k)
                heatmap = cos_map1_list[i][k - 1][0]
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                ax[j].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
            ax[j].axis('off')
            ax[j].scatter(trg_point1_list[i][k - 1][0],
                          trg_point1_list[i][k - 1][1],
                          c=[reds(k / top_k)],
                          s=scatter_size)
            ax[j].set_title('target P1')

            j = k * 2
            ax[j].imshow(imglist[-1])
            if cos_map2_list[0] is not None:
                print("i,j,k: ", i, j, k)
                heatmap = cos_map2_list[i][k - 1][0]
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                ax[j].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
            ax[j].axis('off')
            ax[j].scatter(trg_point2_list[i][k - 1][0],
                          trg_point2_list[i][k - 1][1],
                          c=[blues(k / top_k)],
                          s=scatter_size)
            ax[j].set_title('target P2')

        ax[-1].imshow(trg_mask, cmap='gray')
        ax[-1].axis('off')
        ax[-1].set_title('target mask')
        trg_point1 = np.mean(trg_point1_list[i], axis=0)
        trg_point2 = np.mean(trg_point2_list[i], axis=0)
        ax[-1].scatter(trg_point1[0], trg_point1[1], c=[reds(k / top_k)], s=scatter_size)
        ax[-1].scatter(trg_point2[0], trg_point2[1], c=[blues(k / top_k)], s=scatter_size)

    plt.plot()
    plt.savefig(save_name)
    plt.close()


def nearest_distance_to_mask_contour(mask, x, y, threshold=122, stride=30):
    # Convert the boolean mask to an 8-bit image
    dist_list = []
    last_mask = ((mask > 0).astype(np.uint8) * 255)
    threshold_list = list(range(0, 255, stride)) + [threshold]  # the last one is the threshold value
    for mask_threshold in threshold_list:
        mask_8bit = ((mask > mask_threshold).astype(np.uint8) * 255)
        if mask_8bit.sum() == 0:
            mask_8bit = (mask == mask.max()).astype(np.uint8) * 255
        # Find the contours in the mask
        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Check if point is inside any contour
        num = 0
        y = min(y, np.array(mask).shape[0] - 1)
        x = min(x, np.array(mask).shape[1] - 1)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) == 1:  # Inside contour
                num += 1
        if num % 2 == 1:
            dist_list.append(0)
            continue

        # If point is outside all contours, find the minimum distance between the point and each contour
        min_distance = float('inf')
        for contour in contours:
            distance = cv2.pointPolygonTest(contour, (x, y), True)  # Measure distance
            if abs(distance) < min_distance:
                min_distance = abs(distance)

        # normalize the distance with the diagonal length of the mask
        diag_len = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
        dist_list.append(abs(min_distance) / diag_len)
    nss_value = np.array(mask)[int(y), int(x)]
    thres_dist = dist_list.pop()
    return dist_list, nss_value, thres_dist


def dataset_walkthrough(base_dir,
                        method,
                        model,
                        exp_name,
                        res_dir,
                        cor_cfg={},
                        average_pts=True,
                        visualize=False,
                        mask_threshold=120,
                        top_k=1,
                        top_k_type='max',
                        transpose_types=1,
                        device='cuda'):
    eval_pairs = 0
    total_dists1, nss_values1, thres_dists1, res_trg_points1 = {}, {}, {}, {}
    total_dists2, nss_values2, thres_dists2, res_trg_points2 = {}, {}, {}, {}
    # gt_dir = os.path.join(base_dir, 'GT')
    # base_dir = os.path.join(base_dir, 'egocentric')
    transpose_img_funcs = [
        lambda x: x,
        lambda x: x.rotate(90, expand=True),
        lambda x: x.rotate(180, expand=True),
        lambda x: x.rotate(-90, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(180, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True),
    ]
    # for trg_object in os.listdir(base_dir):
    #     eval_pairs += len(os.listdir(os.path.join(base_dir, trg_object)))
    eval_pairs = len(os.listdir(base_dir))
    print(f'Start evaluating {eval_pairs} correspondance pairs...')

    cor_cfg['device'] = device
    cor_cfg['visualize'] = visualize

    pbar = tqdm(total=eval_pairs)
    if visualize:
        # vis_path = os.path.join()
        if os.path.exists(res_dir):
            confrim = input(f'The result folder {res_dir} already exists. input y to remove it...')
            if confrim == 'y':
                shutil.rmtree(res_dir, ignore_errors=True)
            else:
                exit()
        os.makedirs(res_dir, exist_ok=True)

    for trg_category in os.listdir(base_dir):  #category level
        if trg_category in ['Box']:
            continue
        cur_category_dir = os.path.join(base_dir, trg_category)
        total_dists1[trg_category], nss_values1[trg_category], thres_dists1[trg_category], res_trg_points1[
            trg_category] = [], [], [], {}
        total_dists2[trg_category], nss_values2[trg_category], thres_dists2[trg_category], res_trg_points2[
            trg_category] = [], [], [], {}

        find_mask = False
        print('cur_category_dir:', cur_category_dir)

        for trg_object in os.listdir(cur_category_dir):  # 100900
            object_path = os.path.join(cur_category_dir, trg_object)
            src_images, \
            src_point1_list, trg_point1_list, cor_map1_list, \
            src_point2_list, trg_point2_list, cor_map2_list, \
            src_object_list, src_category_list = [], [], [], [], [], [], [], [],[]

            for instance in os.listdir(object_path):
                instance_path = os.path.join(object_path, instance)  #top x or _rgb.png, or mask.png
                if instance_path.endswith('_rgb.png'):  #origin trg pic
                    trg_image = instance_path
                    continue
                if 'interaction_mask' in instance_path:
                    mask_file = instance_path
                    find_mask = True
                    with Image.open(mask_file) as img:
                        try:
                            trg_mask = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        except:
                            trg_mask = np.array(img)
                    continue

                cur_k = int(instance_path.split('top_')[1])
                if cur_k > top_k:
                    continue
                print('K: ', cur_k, top_k)
                for file in os.listdir(instance_path):  #in topK
                    if file.endswith('.png') and 'top' in file:
                        src_trial_id, src_category, src_shape_id, _, _ = file.split('_')

                        src_image_path = os.path.join(instance_path, file)
                        src_images.append(src_image_path)
                        src_category_list.append(str(src_category))
                        # src_res_path = os.path.join(instance_path)
                    elif file.endswith('.json'):  #result
                        with open(os.path.join(instance_path, file), 'r') as fin:
                            src_result = json.load(fin)
                        y1, x1 = src_result['pixel_locs_1']
                        src_p1 = [x1, y1]
                        y2, x2 = src_result['pixel_locs_2']
                        src_p2 = [x2, y2]

                        src_point1_list.append([src_p1])
                        src_point2_list.append([src_p2])

            pbar.set_description(f'{trg_category}-{trg_object}')
            trg_prompt = f'a photo of {trg_category}'
            imglist, new_src_points1_list, new_src_points2_list, cor_values_list = [], [], [], []
            print('src_imgs:', len(src_images))
            for i in range(len(src_images)):
                src_prompt = f'a photo of a {src_category_list[i]}'
                w, h = Image.open(src_images[i]).size
                transpose_pts_funcs = [
                    lambda x, y: (x, y),
                    lambda x, y: (y, w - x),  # 90 
                    lambda x, y: (w - x, h - y),  # 180
                    lambda x, y: (h - y, x),  # -90
                    lambda x, y: (w - x, y),  # flip
                    lambda x, y: (y, x),  # flip 90
                    lambda x, y: (x, h - y),  # flip 180
                    lambda x, y: (h - y, w - x),  # flip -90
                ]
                trg_pnts1_tmp, src_pnts1_tmp, cor_maps1_tmp, cor_values1_tmp = [], [], [], []
                trg_pnts2_tmp, src_pnts2_tmp, cor_maps2_tmp, cor_values2_tmp = [], [], [], []
                src_image_tmp = []
                # os.makedirs(f'{res_dir}/{trg_object}', exist_ok=True)

                for j in range(transpose_types):
                    trg_pnts1, src_pnts1, cor_maps1, trg_pnts2, src_pnts2, cor_maps2, src_image, cor_values1, cor_values2 = get_dual_cor_pairs(
                        method, model, src_images[i], trg_image, src_point1_list[i], src_point2_list[i], src_prompt,
                        trg_prompt, cor_cfg, transpose_img_funcs[j], transpose_pts_funcs[j], device)

                    trg_pnts1_tmp.append(trg_pnts1)
                    src_pnts1_tmp.append(src_pnts1)
                    cor_maps1_tmp.append(cor_maps1)

                    trg_pnts2_tmp.append(trg_pnts2)
                    src_pnts2_tmp.append(src_pnts2)
                    cor_maps2_tmp.append(cor_maps2)

                    cor_values1_tmp.append(np.mean(cor_values1))
                    cor_values2_tmp.append(np.mean(cor_values2))
                    src_image_tmp.append(src_image)

                dual_cor_values_tmp = [x + y for x, y in zip(cor_values1_tmp, cor_values2_tmp)]

                selected_idx = np.argmax(dual_cor_values_tmp)
                new_src_points1_list.append(src_pnts1_tmp[selected_idx])
                trg_point1_list.append(trg_pnts1_tmp[selected_idx])
                cor_map1_list.append(cor_maps1_tmp[selected_idx])
                new_src_points2_list.append(src_pnts2_tmp[selected_idx])
                trg_point2_list.append(trg_pnts2_tmp[selected_idx])
                cor_map2_list.append(cor_maps2_tmp[selected_idx])

                cor_values_list.append(dual_cor_values_tmp[selected_idx])
                imglist.append(src_image_tmp[selected_idx])

            trg_points1 = np.mean(trg_point1_list, axis=1)
            trg_points2 = np.mean(trg_point2_list, axis=1)
            if top_k_type == 'max':
                trg_point1 = trg_points1[np.argmax(cor_values_list)]
                trg_point2 = trg_points2[np.argmax(cor_values_list)]

            elif top_k_type == 'avg':
                trg_point1 = np.mean(trg_points1, axis=0)
                trg_point2 = np.mean(trg_points2, axis=0)

            trg_dist1, nss_value1, thres_dist1 = nearest_distance_to_mask_contour(trg_mask, trg_point1[0],
                                                                                  trg_point1[1], mask_threshold)
            trg_dist2, nss_value2, thres_dist2 = nearest_distance_to_mask_contour(trg_mask, trg_point2[0],
                                                                                  trg_point2[1], mask_threshold)

            total_dists1[trg_category].append(trg_dist1)
            thres_dists1[trg_category].append(thres_dist1)
            nss_values1[trg_category].append(nss_value1)
            res_trg_points1[trg_category][trg_object] = trg_point1_list

            total_dists2[trg_category].append(trg_dist2)
            thres_dists2[trg_category].append(thres_dist2)
            nss_values2[trg_category].append(nss_value2)
            res_trg_points2[trg_category][trg_object] = trg_point2_list

            # print(trg_point, trg_dist)ipy
            if visualize:
                vis_dir = os.path.join(res_dir, f'{trg_category}-{trg_object}')
                imglist.append(Image.open(trg_image).convert('RGB'))
                os.makedirs(vis_dir, exist_ok=True)

                file_name = f'{trg_category}-{trg_object}_Points_{thres_dist1:.2f}_{nss_value1}'
                if top_k_type == 'max':
                    file_name += f'_max_idx{np.argmax(cor_values_list)}'

                plot_dual_img_pairs(imglist, new_src_points1_list, trg_point1_list, new_src_points2_list,
                                    trg_point2_list, trg_mask, 1, cor_map1_list, cor_map2_list,
                                    os.path.join(vis_dir, file_name + '.png'))

            pbar.update(1)
    pbar.close()
    return total_dists1, nss_values1, thres_dists1, res_trg_points1, total_dists2, nss_values2, thres_dists2, res_trg_points2


def analyze_dists(total_dists, nss_values, thres_dists, res_dir=None):
    all_dists, all_nss, thres_dist, lines = [], [], [], []

    if res_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for trg_object in total_dists.keys():
            all_dists += total_dists[trg_object]
            dist_curve = np.array(total_dists[trg_object]).mean(axis=0)
            sr_curve = (np.array(total_dists[trg_object]) == 0).sum(axis=0) / len(total_dists[trg_object])
            axes[0].plot(dist_curve, label=trg_object)
            axes[1].plot(sr_curve, label=trg_object)

        axes[0].plot(np.array(all_dists).mean(axis=0), label='all', linewidth=3, color='black')
        axes[1].plot((np.array(all_dists) == 0).sum(axis=0) / len(all_dists), label='all', linewidth=3, color='black')
        axes[0].legend()
        axes[1].legend()
        axes[0].set_title('DTM')
        axes[1].set_title('SR')

        plt.savefig(os.path.join(res_dir, 'dist_curve.png'))

    for trg_object in thres_dists.keys():
        all_nss += nss_values[trg_object]
        thres_dist += thres_dists[trg_object]
        lines.append(
            f'{trg_object.split("_")[0]:12s}: dist mean:{np.array(thres_dists[trg_object]).mean():.3f}, nss mean: {np.array(nss_values[trg_object]).mean():.1f}, success rate: {(np.array(thres_dists[trg_object]) == 0).sum() / len(thres_dists[trg_object]):.3f} ({(np.array(thres_dists[trg_object]) == 0).sum()}/{len(thres_dists[trg_object])})'
        )
    lines.append(
        f'=== ALL  ===: dist mean:{np.mean(thres_dist):.3f}, nss mean: {np.array(all_nss).mean():.1f}, success rate: {(np.array(thres_dist)==0).sum() / len(thres_dist):.3f} ({(np.array(thres_dist)==0).sum()}/{len(thres_dist)})'
    )
    if res_dir is not None:
        with open(os.path.join(res_dir, 'total_dists.txt'), 'w') as f:
            f.writelines([line + '\n' for line in lines])
    for line in lines:
        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        '-m',
                        type=str,
                        default='dift',
                        choices=['dift', 'ldm_sc', 'sd_dino', 'dino_vit'],
                        help='method for correspondance')
    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        default='clip_b32_x2',
                        choices=[
                            'clip_b32', 'clib_b32_x0.5', 'clip_b32_x2', 'clip_b16', 'clip_b16_x0.5', 'clip_b16_x2',
                            'clip_b32_lpips', 'clip_b32_lpips_x0.5', 'clip_b32_lpips_x2', 'resnet_50', 'resnet_50_x0.5',
                            'resnet_50_x2'
                        ],
                        help='dataset for affordance memory')

    parser.add_argument('--exp_name', '-e', type=str, default='', help='experiment name')
    parser.add_argument('--base_dir', '-b_dir', type=str, default='', help='base data dir')
    parser.add_argument('--out_dir', '-o_dir', type=str, default='', help='out data dir')

    parser.add_argument('--mask_threshold',
                        '-s',
                        type=int,
                        default=122,
                        help='mask threshold for success rate calculation')
    parser.add_argument('--visualize', '-v', action='store_true', help='visualize the correspondance pairs')
    parser.add_argument('--avg_pts',
                        '-a',
                        action='store_true',
                        help='average the five source points before (True) or after (False) correspondance')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='use the top k retrieved images')
    parser.add_argument('--top_k_type',
                        '-kt',
                        type=str,
                        default='max',
                        choices=['max', 'avg'],
                        help='max: use the top k with max cor value, avg: use the average results of top k')
    parser.add_argument('--transpose_types',
                        '-t',
                        type=int,
                        default=8,
                        help='1: no transpose, 4: rotations, 8: flip and rotations')
    args = parser.parse_args()
    args.avg_pts = False
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    average_pts, visualize = args.avg_pts, args.visualize
    exp_name = args.dataset
    if args.mask_threshold != 122:
        exp_name += '_s' + str(args.mask_threshold)
    if args.top_k != 1:
        exp_name += f'_top{args.top_k}'
        if args.top_k_type == 'max':
            exp_name += '_max'
    if args.avg_pts:
        exp_name += '_avg'
    if args.transpose_types > 1:
        exp_name += '_transpose'
    exp_name = exp_name if len(args.exp_name) == 0 else args.exp_name
    cor_cfg = get_cor_cfg(args.method)

    model = get_model(args.method, cor_cfg, device=device)
    # base_dir = f'datasets/{args.dataset}'
    base_dir = args.base_dir
    res_dir = os.path.join(args.out_dir, f'results/{args.method}/{exp_name}')
    print(f'res_dir: {res_dir}')
    # os.makedirs(res_dir, exist_ok=True)


    total_dists1, nss_values1, thres_dists1, trg_points1, \
        total_dists2, nss_values2, thres_dists2, trg_points2 \
            = dataset_walkthrough(base_dir, args.method, model, exp_name,
                                                                           res_dir, cor_cfg, average_pts, visualize,
                                                                           args.mask_threshold, args.top_k,
                                                                           args.top_k_type, args.transpose_types,
                                                                           device)

    with open(os.path.join(res_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(
            {
                'args': args,
                'total_dists1': total_dists1,
                'nss_values1': nss_values1,
                'thres_dists1': thres_dists1,
                'trg_points1': trg_points1,
                'total_dists2': total_dists2,
                'nss_values2': nss_values2,
                'thres_dists2': thres_dists2,
                'trg_points2': trg_points2,
            }, f)

    analyze_dists(total_dists1, nss_values1, thres_dists1, res_dir)
    analyze_dists(total_dists2, nss_values2, thres_dists2, res_dir)
