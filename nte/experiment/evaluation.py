# -*- coding: utf-8 -*-
"""
| **@created on:** 8/30/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc
from scipy.stats import skew, skewtest
import numpy as np
import torch
from matplotlib import pyplot as plt
import io
import wandb
from PIL import Image
import os
from nte.experiment.utils import send_plt_to_wandb
import multiprocessing
import platform


def custom_auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, supp=0.80):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.ins_cutoff = 1.0
        self.del_cutoff = 0.0
        self.per_supp = supp
        self.per_supp_len = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.ins_cutoff = 0.99
        # self.del_cutoff = 0.10

    def single_run(self, class_0_mean, class_1_mean, time_series_tensor, explanation, enable_wandb, debug,
                   return_results, save_to=None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        time_series_tensor = time_series_tensor.to(self.device)
        X = time_series_tensor.clone()
        TIMESTEPS = len(time_series_tensor)
        # todo Ramesh: Check whether the model give raw prediction?
        pred, c, acts = self.model.evaluate(time_series_tensor.reshape(1, -1).to(self.device))
        orig_pred = pred
        c = c[0]
        n_steps = (TIMESTEPS + self.step - 1) // self.step
        # cla = np.zeros(int(n_steps) + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        # print("Salient Order keys", salient_order)
        # print("Salient order values", explanation[salient_order])
        # print("Explanation", explanation)
        # print("percentile", np.percentile(explanation, 50, axis=1,keepdims=True))
        orig_cl = c.item()
        ret_auc = 0.0

        print('Original Prediction: ', pred.item(), "class", c.item())

        sal_order = {}

        if self.mode == 'del':
            title = 'Deletion Metric'
            ylabel = '% of Time Steps deleted'
            start = time_series_tensor.clone().to('cpu')
            if orig_cl == 0:
                finish = torch.tensor(class_1_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            else:
                finish = torch.tensor(class_0_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
        elif self.mode == 'ins':
            title = 'Insertion Metric'
            ylabel = '% of Time steps inserted'
            if orig_cl == 0:
                start = torch.tensor(class_1_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            else:
                start = torch.tensor(class_0_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            finish = time_series_tensor.clone().to('cpu')
        else:
            raise Exception('error in mode')

        scores = np.zeros(int(n_steps) + 1)

        # minimality of saliency
        sal = explanation.flatten()
        bins = {b: [] for b in np.linspace(0, 1, 225)}
        for s in sal:
            for b in bins.keys():
                if s <= b:
                    bins[b].append(s)
                    break
        vs = [len(b) / len(sal) for b in bins.values()]

        for i in range(n_steps + 1):
            print(f"{self.mode} -  Step: {i}/{n_steps + 1} ({i / (n_steps + 1) * 100:.2f}%)")
            pred, cl, acts = self.model.evaluate(start.reshape(1, -1).to(self.device))
            scores[i] = acts[0][orig_cl].item()
            # cla[i] = cl.item()
            if self.mode == 'del':
                if (scores[i] >= 0.95):
                    self.per_supp_len[0] = i + 1
                if (scores[i] >= 0.90):
                    self.per_supp_len[1] = i + 1
                if (scores[i] >= 0.80):
                    self.per_supp_len[2] = i + 1
                if (scores[i] >= 0.70):
                    self.per_supp_len[3] = i + 1
                if (scores[i] >= 0.60):
                    self.per_supp_len[4] = i + 1
                if (scores[i] >= 0.50):
                    self.per_supp_len[5] = i + 1
                if (scores[i] >= 0.40):
                    self.per_supp_len[6] = i + 1
                if (scores[i] >= 0.30):
                    self.per_supp_len[7] = i + 1
                if (scores[i] >= 0.20):
                    self.per_supp_len[8] = i + 1
                if (scores[i] >= 0.10):
                    self.per_supp_len[9] = i + 1

            if debug:
                print(
                    f"Acts:{acts}  Pred:{pred}  Orig_pred:{orig_pred}  Cl:{cl}  Orig_cl:{orig_cl}  Scores:{scores[i]}")
            # scores[i] = pred[c]

            if debug or save_to:
                # if(len(np.arange(i+1)) != len(vs[:i+1])):
                #     continue
                plt.figure(figsize=(10, 5))
                plt.subplot(131)
                plt.title('{} {:.1f}%, P(1)={:.2f}'.format(ylabel, 100 * (i - 1) / n_steps, scores[i]))
                plt.plot(list(range(len(X))), X, label="Raw Pattern", color="red", alpha=0.4)
                # plt.plot(list(range(len(start))), start,label="Perturbed Pattern")
                plt.scatter(list(sal_order.keys()), list(sal_order.values()), color="orange", label="Saliency")
                plt.xlabel("Timesteps")
                plt.ylabel("Values")
                plt.legend()

                plt.subplot(132)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1], label="AUC")
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title + f" AUC - {auc(np.arange(i + 1) / n_steps, scores[:i + 1]) if i > 5 else 0.0:.4f}")
                plt.xlabel(ylabel)
                plt.ylabel("Probability")
                plt.title(title + f" AUC - {auc(np.arange(i + 1) / n_steps, scores[:i + 1]) if i > 5 else 0.0:.4f}")
                # plt.text(x=0.4, y=0.95, s=f'AUC: {auc(scores[:i + 1]):.2f}')
                plt.legend()

                plt.subplot(133)
                plt.plot(np.arange(i + 1) / n_steps, vs[:i + 1])
                plt.fill_between(list(bins.keys())[:i + 1], 0, vs[:i + 1], alpha=0.4)
                plt.yscale("log")
                plt.xlabel("% of Time Steps")
                plt.ylabel("Saliency")
                plt.title(f"% Saliency AUC - {auc(list(bins.keys())[:i + 1], vs[:i + 1]) if i > 5 else 0.0:.4f}")
                if save_to:
                    plt.tight_layout()
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    if enable_wandb:
                        wandb.log({f"Evaluation  - {title}": plt})
                        wandb.log({f"{title} scores": float(scores[i]),
                                   f"{title} scores ms": float(vs[i])})
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)][0]
                sal_order[coords[0]] = explanation[int(coords[0])]
                start.cpu().numpy().reshape(TIMESTEPS)[coords] = finish.cpu().numpy().reshape(TIMESTEPS)[coords]

        if self.mode == 'del':
            # deletion game metrics
            print('Percent Suppression Length : ', self.per_supp_len)
            plt.figure()
            plt.title("Percent Suppression Score")
            # for x in range(1,10,1):
            #     plt.bar(str(x*10)+"%", height=self.per_supp_len[x], color=np.random.rand(3), align='center')
            plt.bar("10%", height=self.per_supp_len[0], color=np.random.rand(3), align='center')
            plt.bar("20%", height=self.per_supp_len[1], color=np.random.rand(3), align='center')
            plt.bar("30%", height=self.per_supp_len[2], color=np.random.rand(3), align='center')
            plt.bar("40%", height=self.per_supp_len[3], color=np.random.rand(3), align='center')
            plt.bar("50%", height=self.per_supp_len[4], color=np.random.rand(3), align='center')
            plt.bar("60%", height=self.per_supp_len[5], color=np.random.rand(3), align='center')
            plt.bar("70%", height=self.per_supp_len[6], color=np.random.rand(3), align='center')
            plt.bar("80%", height=self.per_supp_len[7], color=np.random.rand(3), align='center')
            plt.bar("90%", height=self.per_supp_len[8], color=np.random.rand(3), align='center')
            plt.bar("95%", height=self.per_supp_len[9], color=np.random.rand(3), align='center')
            # plt.bar(1,height=self.per_supp_len,color='b',width60.1)
            plt.xlabel("Percent of model output suppression")
            plt.ylabel("TimeSteps")
            plt.legend()
            if save_to:
                plt.savefig(save_to + '_supp.png')
                if enable_wandb:
                    wandb.log({f"Deletion Game": [send_plt_to_wandb(plt, 'Deletion Game')]})
                plt.close()
            else:
                plt.show()
        return_results[self.mode] = [scores, np.array(vs), ret_auc, self.per_supp_len]
        return return_results


def qm_plot(X, model, saliency, class_0_mean, class_1_mean, enable_wandb, save_dir, supp=0.08, debug=False,
            multi_process=True):
    if 'Darwin' in platform.platform():
        multiprocessing.set_start_method('forkserver', force=True)
        multi_process = False
    manager = multiprocessing.Manager()

    process_results = manager.dict()
    TIMESTEPS = len(X)
    STEPS = 20
    # STEPS = 1
    # noise = torch.tensor(generate_gaussian_noise(X, snrdb=0.001), dtype=torch.float32)
    noise = torch.tensor(np.random.random(size=TIMESTEPS), dtype=torch.float32)

    # noise = 0.01
    blur = lambda x: x * noise
    eval_metrics = {}
    insertion = CausalMetric(model, 'ins', STEPS, substrate_fn=torch.zeros_like, supp=supp)
    deletion = CausalMetric(model, 'del', STEPS, substrate_fn=torch.zeros_like, supp=supp)

    if multi_process:
        p1 = multiprocessing.Process(target=insertion.single_run, args=(class_0_mean, class_1_mean,
                                                                        torch.tensor(X, dtype=torch.float32),
                                                                        saliency, enable_wandb, debug, process_results))
        p2 = multiprocessing.Process(target=deletion.single_run, args=(class_0_mean, class_1_mean,
                                                                       torch.tensor(X, dtype=torch.float32),
                                                                       saliency, enable_wandb, debug, process_results))
        p1.start()
        p2.start()
        p1.join()
        scores, ms, ins_ret_auc, ips_len = process_results['ins']
    else:
        scores, ms, ins_ret_auc, ips_len = insertion.single_run(class_0_mean, class_1_mean,
                                                                torch.tensor(X, dtype=torch.float32),
                                                                saliency, enable_wandb, debug, process_results)['ins']

    trapauc = auc(x=list(range(len(scores))), y=scores) / ((len(scores) - 1))
    cauc = custom_auc(scores)

    trapauc_ms = auc(x=list(range(len(ms))), y=ms) / ((len(ms) - 1))
    cauc_ms = custom_auc(ms)
    print(f'Insertion - TrapAUC: {trapauc: .2f} | DiagAUC: {cauc: .2f}')

    stest = skewtest(scores)
    stest_ms = skewtest(ms)
    eval_metrics['Insertion'] = {"trap": trapauc,
                                 "custom": cauc,
                                 "skew": skew(scores),
                                 "pval": float(stest.pvalue),
                                 "zscore": float(stest.statistic),
                                 "scores": scores.tolist(),
                                 "trap_ms": trapauc_ms,
                                 "custom_ms": cauc_ms,
                                 "skew_ms": skew(ms),
                                 "pval_ms": float(stest_ms.pvalue),
                                 "zscore_ms": float(stest_ms.statistic),
                                 "scores_ms": ms.tolist(),
                                 }

    # os.system(f'mkdir -p {save_dir}/metrics/Deletion')
    if multi_process:
        p2.join()
        scores, ms, ins_ret_auc, dps_len = process_results['del']
    else:
        scores, ms, ins_ret_auc, dps_len = deletion.single_run(class_0_mean, class_1_mean,
                                                               torch.tensor(X, dtype=torch.float32),
                                                               saliency, enable_wandb, debug, process_results)['del']
    trapauc = auc(x=list(range(len(scores))), y=scores) / ((len(scores) - 1))
    cauc = custom_auc(scores)

    trapauc_ms = auc(x=list(range(len(ms))), y=ms) / ((len(ms) - 1))
    cauc_ms = custom_auc(ms)

    stest = skewtest(scores)
    stest_ms = skewtest(ms)

    eval_metrics['Deletion'] = {"trap": trapauc,
                                "custom": cauc,
                                "skew": skew(scores),
                                "pval": float(stest.pvalue),
                                "zscore": float(stest.statistic),
                                "scores": scores.tolist(),
                                "trap_ms": trapauc_ms,
                                "custom_ms": cauc_ms,
                                "skew_ms": skew(ms),
                                "pval_ms": float(stest_ms.pvalue),
                                "zscore_ms": float(stest_ms.statistic),
                                "scores_ms": ms.tolist(),
                                }
    eval_metrics['Final'] = {'AUC': eval_metrics['Insertion']['trap'] - eval_metrics['Deletion']['trap'],
                             "ms_sum":float(np.sum(saliency)),
                             "ms_var":float(np.var(saliency))}
    # eval_metrics["ms_sum"]= np.sum(saliency)
    # eval_metrics["ms_var"]= np.var(saliency)
    percs = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
    eval_metrics['Deletion Game'] = {k: v for k, v in zip(percs, dps_len)}
    print(f'Deletion - TrapAUC: {trapauc: .2f} | DiagAUC: {cauc: .2f}')
    print(eval_metrics['Final'])
    return eval_metrics  # ins_ret_auc, del_ret_auc, ips_len, dps_len


# ["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance"])
def run_evaluation_metrics(EVAL_REPLACEMENT, dataset, original_signal, model, mask, SAVE_DIR, ENABLE_WANDB, multi_process=True):
    if EVAL_REPLACEMENT == 'zeros':
        class_0_mean = np.repeat(0, len(dataset.test_class_0_mean))
        class_1_mean = np.repeat(0, len(dataset.test_class_1_mean))
    elif EVAL_REPLACEMENT == 'class_mean':
        class_0_mean = dataset.test_class_0_mean
        class_1_mean = dataset.test_class_1_mean
    elif EVAL_REPLACEMENT == 'instance_mean':
        ins_mean = np.mean(original_signal.cpu().detach().numpy()).repeat(
            original_signal.shape[0])
        class_0_mean = ins_mean
        class_1_mean = ins_mean
    else:
        raise Exception("Unsupported eval replacement")
    return qm_plot(model=model, X=original_signal, saliency=mask,
                   class_0_mean=class_0_mean,
                   class_1_mean=class_1_mean,
                   save_dir=SAVE_DIR, enable_wandb=ENABLE_WANDB, multi_process=multi_process)
