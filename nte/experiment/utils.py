# -*- coding: utf-8 -*-
"""
| **@created on:** 7/23/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from torchvision import models
import numpy as np
from torch.autograd import Variable
import torch
import cv2
import wandb
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc
from scipy.stats import skew, skewtest
import os
import argparse
from nte import NTE_TRAINED_MODEL_PATH
import io
from PIL import Image
from nte.utils.plot_utils import plot_saliency_cmap_multi
import matplotlib.pyplot as plt
from nte.utils import normalize
import seaborn as sns
from matplotlib.colors import ListedColormap
from nte.data.synth.blipv3.blipv3_dataset import BlipV3Dataset
from nte.data.real.wafer.wafer import WaferDataset
from nte.data.real.gun_point.GunPointDataset import GunPointDataset
from nte.data.real.ford_a.FordADataset import FordADataset
from nte.data.real.ford_b.FordBDataset import FordBDataset
from nte.data.real.earthquakes.EarthquakesDataset import EarthquakesDataset
from nte.data.real.ptb_heart_rate.ptb_heart_rate import PTBHeartRateDataset
from nte.data.real.ecg.EcgDataset import EcgDataset
from nte.data.real import ComputersDataset, CricketXDataset
import math
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

SNS_CMAP = ListedColormap(sns.light_palette('red').as_hex())


def dual_min_max_norm(data, fixed_min=None, fixed_max=None):
    pos_indices = np.argwhere(data > 0)
    pos_features = data[pos_indices]
    neg_indices = np.argwhere(data < 0)
    neg_features = data[neg_indices]

    pos_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    pos_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    pos_features = (pos_features - pos_features_min) / (pos_features_max - pos_features_min)

    neg_features = np.abs(neg_features)
    neg_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    neg_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    neg_features = (neg_features - neg_features_min) / (neg_features_max - neg_features_min)

    data[pos_indices] = pos_features
    data[neg_indices] = -neg_features
    return data


def print_var_stats(var):
    print(f"Min: {var.min()} ({np.argmin(var)}) | Max: {var.max()} ({np.argmax(var)}) | Var: {var.var()}")

def distance_metrics(sample_a, sample_b, def_key=''):
    dist_metrics = {}
    dist_metrics['euc'] = euclidean(sample_a, sample_b)
    dist_metrics['dtw'] = fastdtw(sample_a, sample_b)[0]
    dist_metrics['cs'] = cosine_similarity([sample_a], [sample_b])[0][0]
    return {def_key + k: v for k, v in dist_metrics.items()}


def model_metrics(model, sample, label, def_key=''):
    model_metrics_data = {}
    raw_preds = model(torch.tensor(sample, dtype=torch.float32))
    prob = torch.nn.Softmax(dim=-1)(raw_preds).numpy()
    raw_preds = raw_preds.numpy()
    model_metrics_data['label'] = label
    model_metrics_data['raw_pred_class_0'] = float(raw_preds[0])
    model_metrics_data['raw_pred_class_1'] = float(raw_preds[1])
    model_metrics_data['prob_class_0'] = float(prob[0])
    model_metrics_data['prob_class_1'] = float(prob[1])
    model_metrics_data['conf'] = np.max(prob)
    model_metrics_data['prediction'] = np.argmax(prob)
    model_metrics_data['pred_acc'] = 1 if model_metrics_data['prediction'] == label else 0
    return {def_key + k: v for k, v in model_metrics_data.items()}


def replacement_sample_config(xt_index, rt_index, model, dataset, dataset_type):
    xt = dataset.__getattribute__(dataset_type + '_data')[xt_index]
    rt = dataset.__getattribute__(dataset_type + '_data')[rt_index]
    replacement_config = distance_metrics(xt, rt, def_key='s_dist_')
    replacement_config = {**replacement_config,
                          **distance_metrics(xt, dataset.test_class_0_mean, def_key='m_xt_dist_0_')}
    replacement_config = {**replacement_config,
                          **distance_metrics(xt, dataset.test_class_1_mean, def_key='m_xt_dist_1_')}
    replacement_config = {**replacement_config,
                          **distance_metrics(rt, dataset.test_class_0_mean, def_key='m_rt_dist_0_')}
    replacement_config = {**replacement_config,
                          **distance_metrics(rt, dataset.test_class_1_mean, def_key='m_rt_dist_1_')}

    if dataset.__getattribute__(dataset_type + '_label')[xt_index] == 0:
        replacement_config = {**replacement_config,
                              **distance_metrics(xt, dataset.test_class_0_mean, def_key='m_xt_dist_')}
        replacement_config = {**replacement_config,
                              **distance_metrics(rt, dataset.test_class_0_mean, def_key='m_rt_dist_')}
        replacement_config = {**replacement_config,
                              **distance_metrics(xt, dataset.test_class_1_mean, def_key='m_xt_dist_o')}
        replacement_config = {**replacement_config,
                              **distance_metrics(rt, dataset.test_class_1_mean, def_key='m_rt_dist_o_')}

    replacement_config = {**replacement_config,
                          **distance_metrics(rt, dataset.test_class_1_mean, def_key='m_rt_dist')}
    replacement_config = {**replacement_config,
                          **model_metrics(model, xt, label=dataset.__getattribute__(dataset_type + '_label')[xt_index],
                                          def_key='xt_')}
    replacement_config = {**replacement_config,
                          **model_metrics(model, rt, label=dataset.__getattribute__(dataset_type + '_label')[rt_index],
                                          def_key='rt_')}
    replacement_config['rt_class'] = 0 if replacement_config['rt_label'] == replacement_config[
        'xt_label'] else 1
    return replacement_config

default_model_paths_rnn = {
    "wafer": "wafer_rnn_ce.ckpt",
    "earthquakes": "earthquakes_rnn_ce.ckpt",
    "ford_a": "ford_a_rnn_ce.ckpt",
    "ford_b": "ford_b_rnn_ce.ckpt",
    "computers": "computers_rnn_ce.ckpt",
    "gun_point": "gun_point_rnn_ce.ckpt",
    "cricket_x": "cricket_x_rnn_ce.ckpt",
    "ecg": "ecg_rnn_ce.ckpt",
    "ptb": "ptb_heart_rate_rnn_ce.ckpt"
}


default_model_paths_dnn = {
    "wafer": "wafer_dnn_ce.ckpt",
    "earthquakes": "earthquakes_dnn_ce.ckpt",
    "ford_a": "ford_a_dnn_ce.ckpt",
    "ford_b": "ford_b_dnn_ce.ckpt",
    "computers": "computers_dnn_ce.ckpt",
    "gun_point": "gun_point_dnn_ce.ckpt",
    "cricket_x": "cricket_x_dnn_ce.ckpt",
    "ecg": "ecg_dnn_ce.ckpt",
    "ptb": "ptb_heart_rate_dnn_ce.ckpt"
}


def set_global_seed(seed_value):
    print(f"Setting seed ({seed_value})  . . .")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)
    random.seed(seed_value)


def get_run_configuration(args, dataset, TASK_ID):
    if args.dataset_type == 'train':
        data = dataset.train_data
        label = dataset.train_label
    elif args.dataset_type == 'test':
        data = dataset.test_data
        label = dataset.test_label
    elif args.dataset_type == 'valid':
        data = dataset.valid_data
        label = dataset.valid_label
    else:
        raise Exception(f"Unknown dataset_type : {args.dataset_type}. Supported - [train, test, representative]")
    print(f"Running on {args.dataset_type} data")

    if args.run_mode == 'single':
        ds = enumerate(zip([data[args.single_sample_id]], [label[args.single_sample_id]]))
        print(f"Running a single sample: idx {args.single_sample_id} . . .")
    elif args.run_mode == 'local':
        ds = enumerate(zip(data, label))
        print(f"Running in local mode on complete data . . .")
    else:
        print(f"Running in turing mode using slurm tasks . . .")
        if args.jobs_per_task > 0:
            args.samples_per_task = math.ceil(len(data) / args.jobs_per_task)
        ds = enumerate(
            zip(data[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + args.samples_per_task],
                label[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + args.samples_per_task]))
    return ds


def backgroud_data_configuration(BACKGROUND_DATA, BACKGROUND_DATA_PERC, dataset):
    # Background Data Configuration
    if BACKGROUND_DATA == 'train':
        print("Using TRAIN data as background data")
        bg_data = dataset.train_data
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    elif BACKGROUND_DATA == 'test':
        print("Using TEST data as background data")
        bg_data = dataset.test_data
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    else:
        print("Using Instance as background data (No BG Data)")
        bg_data = dataset.test_data
        bg_len = 0
    return bg_data, bg_len


number_to_dataset = {
    "1": "wafer",
    "2": "cricket_x",
    "3": "gun_point",
    "4": "earthquakes",
    "5": "computers",
    "6": "ford_a",
    "7": "ford_b",
    "8": "ptb",
    "9": "ecg"
}


def dataset_mapper(DATASET):
    # Dataset Mapper
    if DATASET == 'blip':
        dataset = BlipV3Dataset()
    elif DATASET in ['wafer', "1"]:
        dataset = WaferDataset()
    elif DATASET in ['cricket_x', "2"]:
        dataset = CricketXDataset()
    elif DATASET in ['gun_point', "3"]:
        dataset = GunPointDataset()
    elif DATASET in ['earthquakes', "4"]:
        dataset = EarthquakesDataset()
    elif DATASET in ['computers', "5"]:
        dataset = ComputersDataset()
    elif DATASET in ['ford_a', "6"]:
        dataset = FordADataset()
    elif DATASET in ['ford_b', "7"]:
        dataset = FordBDataset()
    elif DATASET in ['ptb', "8"]:
        dataset = PTBHeartRateDataset()
    elif DATASET in ['ecg', "9"]:
        dataset = EcgDataset()
    else:
        raise Exception(f"Unknown Dataset: {DATASET}")
    return dataset


def send_plt_to_wandb(plt, title):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return wandb.Image(Image.open(buf), caption=title)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image(img):
    if len(img.shape) == 4:
        img = np.transpose(img[0], (1, 2, 0))
        return np.uint8(255 * img)
    else:
        return np.uint8(255 * img)


# def tv_norm(input, tv_beta):
#     img = input[0, 0, :]
#     row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
#     col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
#     return row_grad + col_grad

def tv_norm(signal, tv_beta):
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


def preprocess_image(img, use_cuda):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save_timeseries(mask, time_series, save_dir, dataset, algo,blurred=None , enable_wandb=False, raw_mask=None, category=None):
    mask = mask
    # mask = (mask - np.min(mask)) / (np.max(mask) + 1e-8)
    # uplt = plot_saliency_cmap(data=time_series, weights=mask, plt=plt, display=True, dataset_name=dataset, labels=algo)
    if raw_mask is None:
        uplt = plot_cmap(time_series, mask)
    else:
        uplt = plot_cmap_multi(time_series, norm_saliency=mask, raw_saliency=raw_mask, category=category)
    uplt.xlabel("Timesteps")
    uplt.ylabel("Value")
    if enable_wandb:
        wandb.log({"Result": [send_plt_to_wandb(uplt, 'Saliency Visualization')]})


def save(mask, img, blurred, save_dir, enable_wandb):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite(f"{save_dir}/res-perturbated.png", np.uint8(255 * perturbated))

    if enable_wandb:
        wandb.log({"Result": [wandb.Image(np.uint8(255 * perturbated), caption="Perurbation"),
                              wandb.Image(np.uint8(255 * heatmap), caption="Heatmap"),
                              wandb.Image(np.uint8(255 * mask), caption='Mask'),
                              wandb.Image(np.uint8(255 * cam), caption='CAM')]})
    cv2.imwrite(f"{save_dir}/res-heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(f"{save_dir}/res-mask.png", np.uint8(255 * mask))
    cv2.imwrite(f"{save_dir}/res-cam.png", np.uint8(255 * cam))


def numpy_to_torch(img, use_cuda, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model(use_cuda):
    model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    return model


def get_model(dataset, dest_path=None, model=None, loss=None, lr=None, epoch=None, use_cuda=False, bbm='dnn'):
    if dest_path is None:
        if None in [dataset, model, loss, lr, epoch]:
            raise Exception("Either provide dest path or dataset & model & loss & lr &  epoch")
        path = f"{NTE_TRAINED_MODEL_PATH}/all_bbms/{dataset}_model_{model}_loss_{loss}_lr_{lr}_epoch_{epoch}.ckpt"
    elif dest_path == 'default':
        if bbm == 'dnn':
            path = f"{NTE_TRAINED_MODEL_PATH}/all_bbms/{default_model_paths_dnn[dataset]}"
        else:
            path = f"{NTE_TRAINED_MODEL_PATH}/all_bbms/{default_model_paths_rnn[dataset]}"
    else:
        path = f"{NTE_TRAINED_MODEL_PATH}/all_bbms/{dest_path}"

    model = torch.load(path)
    model.eval()
    if use_cuda:
        model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    return model

def plot_cmap_multi(data, norm_saliency, raw_saliency, category):
    CMAP = ListedColormap([*sns.light_palette('red').as_hex()[::-1], "#FFFFFF" , *sns.light_palette('green').as_hex()])
    try:
        data = data#.cpu().detach().numpy().flatten().tolist()
        raw_saliency = raw_saliency if category==0 else -raw_saliency
        # raw_saliency = dual_min_max_norm(raw_saliency, fixed_max=1.0, fixed_min=0.0)
        timesteps = len(data)
        plt.clf()
        fig = plt.gcf()

        raw_saliency[np.argmin(raw_saliency)]=-1
        raw_saliency[np.argmax(raw_saliency)]=1
        im = plt.imshow(raw_saliency.reshape([1, -1]), cmap=CMAP, aspect="auto", alpha=0.85,
                        extent=[0, len(raw_saliency) - 1, float(np.min([np.min(data)])) - 1e-1,
                                float(np.max([np.max(data)])) + 1e-1]
                        )
        plt.plot(data, lw=4)
        plt.grid(False)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=4)
    except Exception as e:
        print(e)
        print("Failed to generate the CMAP!")
        pass
    return plt

def plot_cmap(data, saliency):
    try:
        data = data#.cpu().detach().numpy().flatten().tolist()
        timesteps = len(data)
        plt.clf()
        fig = plt.gcf()
        im = plt.imshow(saliency.reshape([1, -1]), cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                        extent=[0, len(saliency) - 1, float(np.min([np.min(data), np.min(saliency)])) - 1e-1,
                                float(np.max([np.max(data), np.max(saliency)])) + 1e-1]
                        )
        plt.plot(data, lw=4)
        plt.grid(False)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=4)
    except Exception:
        print("Failed to generate the CMAP!")
        pass
    return plt


