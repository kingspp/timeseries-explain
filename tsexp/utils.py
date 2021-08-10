import numpy as np
from torch.autograd import Variable
import torch
import wandb
from PIL import Image
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import io
import logging

logger = logging.getLogger(__name__)

SNS_CMAP = ListedColormap(sns.light_palette('red').as_hex())


def send_plt_to_wandb(plt, title):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return wandb.Image(Image.open(buf), caption=title)


def tv_norm(signal, tv_beta):
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


def save_timeseries(mask, time_series, enable_wandb=False, raw_mask=None, category=None):
    mask = mask
    if raw_mask is None:
        uplt = plot_cmap(time_series, mask)
    else:
        uplt = plot_cmap_multi(time_series, norm_saliency=mask, raw_saliency=raw_mask, category=category)
    uplt.xlabel("Timesteps")
    uplt.ylabel("Value")
    if enable_wandb:
        wandb.log({"Result": [send_plt_to_wandb(uplt, 'Saliency Visualization')]})


def numpy_to_torch(img, use_cuda, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    return Variable(output, requires_grad=requires_grad)


def plot_cmap_multi(data, raw_saliency, category):
    CMAP = ListedColormap([*sns.light_palette('red').as_hex()[::-1], "#FFFFFF",
                           *sns.light_palette('green').as_hex()])
    try:
        data = data.cpu().detach().numpy().flatten().tolist()
        raw_saliency = raw_saliency if category == 0 else -raw_saliency

        plt.clf()
        fig = plt.gcf()

        raw_saliency[np.argmin(raw_saliency)] = -1
        raw_saliency[np.argmax(raw_saliency)] = 1
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
        plt.savefig("/tmp/fig.png", dpi=400)
    except Exception as e:
        logger.error(f"Failed to generate the CMAP!: \n{e}")
    return plt


def plot_cmap(data, saliency):
    try:
        data = data.cpu().detach().numpy().flatten().tolist()
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
    except Exception as e:
        logger.error(f"Failed to generate the CMAP!: \n{e}")
    return plt
