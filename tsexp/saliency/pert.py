# -*- coding: utf-8 -*-
"""
| **@created on:** 9/28/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""
import torch
import cv2
import numpy as np
import wandb
from pert.utils import tv_norm, numpy_to_torch, save_timeseries
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import logging
import random
from nte.experiment.evaluation import run_evaluation_metrics
from nte.utils.perturbation_manager import PerturbationManager
from nte.models.saliency_model import Saliency
from torch.optim.lr_scheduler import ExponentialLR
from nte.utils.priority_buffer import PrioritizedBuffer

logger = logging.getLogger(__name__)


class PertSaliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(PertSaliency, self).__init__(background_data=background_data,
                                           background_label=background_label,
                                           predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.r_index = random.randrange(0, len(self.background_data)) if self.args.r_index < 0 else self.args.r_index
        self.rs_priority_buffer = None
        self.ro_priority_buffer = None
        self.eps = 1.0
        self.eps_decay = 0.9991

    def priority_dual_greedy_pick_rt(self, kwargs, data, label):
        self.eps *= self.eps_decay
        if np.random.uniform() < self.eps:
            self.mode = 'Explore'
            rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
            ro_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{1-int(label)}_data")))]
            Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]], [1.0]
            Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{1-int(label)}_data")[ro_index[0]]], [1.0]
        else:
            self.mode = 'Exploit'
            Rs, rs_weight, rs_index = self.rs_priority_buffer.sample(1)
            Ro, ro_weight, ro_index = self.ro_priority_buffer.sample(1)
        return {'rs': [Rs, rs_weight, rs_index],
                'ro': [Ro, ro_weight, ro_index]}

    def dynamic_dual_pick_zt(self, kwargs, data, label):
        ds = kwargs['dataset'].test_class_0_indices
        ls = len(ds)
        ods = kwargs['dataset'].test_class_1_indices
        ols = len(ods)
        self.r_index = random.randrange(0, ls)
        self.ro_index = random.randrange(0, ols)
        Zt = self.background_data[ds[self.r_index]]
        ZOt = self.background_data[ods[self.ro_index]]
        return Zt, ZOt

    def dynamic_pick_zt(self, kwargs, data, label):
        self.r_index = None
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                self.r_index = random.randrange(0, len(self.background_data))
                Zt = torch.tensor(self.background_data[self.r_index],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    sds = kwargs['dataset'].test_class_0_indices
                    sls = len(sds)
                else:
                    sds = kwargs['dataset'].test_class_1_indices
                    sls = len(sds)
                self.r_index = random.randrange(0, sls)
                Zt = torch.tensor(sds[self.r_index], dtype=torch.float32)
        return Zt

    def static_pick_zt(self, kwargs, data, label):
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                Zt = torch.tensor(self.background_data[self.r_index], dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][0],
                                      dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][1],
                                      dtype=torch.float32)
        return Zt

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def generate_saliency(self, data, label, **kwargs):

        self.rs_priority_buffer = PrioritizedBuffer(
            background_data=getattr(kwargs['dataset'], f"test_class_{int(label)}_data"))
        self.ro_priority_buffer = PrioritizedBuffer(
            background_data=getattr(kwargs['dataset'], f"test_class_{1 - int(label)}_data"))
        self.eps = 1.0

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        category = np.argmax(kwargs['target'].cpu().data.numpy())

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.cpu().detach().numpy().flatten(),
                algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                original_label=label, sample_id=self.args.single_sample_id)

        plt.plot(data, label="Original Signal Norm")
        gkernel = cv2.getGaussianKernel(3, 0.5)
        gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
        plt.plot(gaussian_blur_signal, label="Gaussian Blur")
        median_blur_signal = median_filter(data, 3)
        plt.plot(median_blur_signal, label="Median Blur")
        blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
        plt.plot(blurred_signal, label="Blurred Signal")
        mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
        blurred_signal_norm = blurred_signal / np.max(blurred_signal)
        plt.plot(blurred_signal_norm, label="Blur Norm")

        if self.enable_wandb:
            wandb.log({'Initialization': plt}, step=0)

        blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
        mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)

        optimizer = torch.optim.Adam([mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        print(f"{self.args.algo}: Category with highest probability {category}")
        print(f"{self.args.algo}: Optimizing.. ")

        metrics = {"TV Norm": [], 'TV Coeff': [], "MSE": [], "Budget": [], "Total Loss": [],
                   "Confidence": [],
                   "Saliency Var": [],
                   "Saliency Sum": [], "MSE Coeff": [], "L1 Coeff": [], "Mean Gradient": [],
                   "Category": [],
                   "epoch": {}, 'MSE Var': [], 'DIST': []}
        mse_loss_fn = torch.nn.MSELoss(reduction='mean')

        original_class_predictions = torch.max(kwargs['target']).item()

        for i in (range(self.args.max_itr)):
            CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'
            # os.system(f"mkdir -p {CUR_DIR}")

            picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=label)
            rs, rs_weight, rs_index = picks['rs']
            ro, ro_weight, ro_index = picks['ro']
            Rt = []
            Rm = []

            for e, (rs_e, ro_e, m) in enumerate(zip(rs[0].flatten(), ro[0].flatten(), mask.detach().numpy().flatten())):
                if m < 0:
                    Rt.append(rs_e)
                    Rm.append("z")
                else:
                    Rt.append(ro_e)
                    Rm.append("o")
            Rt = torch.tensor(Rt, dtype=torch.float32)
            upsampled_mask = (mask)

            perturbated_input = data.mul(upsampled_mask) + Rt.mul(1 - upsampled_mask)

            if self.args.enable_blur:
                perturbation = blurred_signal.mul(upsampled_mask)
                perturbated_input += perturbation

            if self.args.enable_noise:
                noise = np.zeros(data.shape, dtype=np.float32)
                cv2.randn(noise, 0, 0.3)
                noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
                perturbated_input += noise
            perturbated_input = perturbated_input.flatten()

            with torch.no_grad():
                outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape([1, -1])))[0]

            metrics['Confidence'].append(float(outputs[category].item()))

            if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                self.perturbation_manager.add_perturbation(
                    perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                    step=i, confidence=metrics['Confidence'][-1])
            if self.args.bbm == 'rnn':
                c1 = self.args.mse_coeff * mse_loss_fn(outputs[1], kwargs['target'].squeeze(0)[1])
            else:
                c1 = self.args.mse_coeff * mse_loss_fn(outputs[1], kwargs['target'][1])
            c2 = self.args.l1_coeff * torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            c3 = self.args.tv_coeff * tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0

            loss = (c1 + c2 + c3)
            rs_prios = loss * (rs_weight[0])
            ro_prios = loss * (ro_weight[0])
            loss = loss * (rs_weight[0] + ro_weight[0]) / 2
            optimizer.zero_grad()
            loss.backward()
            metrics['Mean Gradient'].append(float(np.mean(mask.grad.cpu().detach().numpy())))
            metrics['TV Norm'].append(float(c3.item()))
            metrics['MSE'].append(float(c1.item()))
            metrics['Budget'].append(float(c2.item()))
            metrics['Total Loss'].append(float(loss.item()))
            metrics['Category'].append(int(np.argmax(outputs.cpu().detach().numpy())))
            metrics['Saliency Sum'].append(float(np.sum(mask.cpu().detach().numpy())))
            metrics['Saliency Var'].append(float(np.var(mask.cpu().detach().numpy())))
            metrics['MSE Coeff'].append(self.args.mse_coeff)
            metrics['TV Coeff'].append(self.args.tv_coeff)
            metrics['L1 Coeff'].append(self.args.l1_coeff)
            metrics['MSE Var'].append(mse_var)

            if self.args.early_stopping:
                if i > self.args.early_stop_criteria_patience and metrics['Confidence'][-1] <= (
                        self.args.early_stop_criteria_perc / 100) * original_class_predictions:
                    logger.debug(
                        f"Early Stop Criteria (Patience: {self.args.early_stop_criteria_patience}, Perc: {self.args.early_stop_criteria_perc}) met - Step: {i} | Confidence: {metrics['Confidence'][-1]}")
                    break

            logger.debug(
                f"Iter: {i}/{self.args.max_itr} | ({i / self.args.max_itr * 100:.2f}%)"
                f"| MSE: {metrics['MSE'][-1]:.4f}, V:{mse_var:.4f}"
                f"| B: {metrics['Budget'][-1]:.4f} | TL: {metrics['Total Loss'][-1]:.4f} "
                f"| S:{metrics['Saliency Sum'][-1]:.4f}"
                f"| RSI: {rs_index[0]} WSI: {rs_weight[0]:.4f}",
                f"| ROI: {ro_index[0]} WOI: {ro_weight[0]:.4f}"
                f"| EPS: {self.eps:.2f} | M: {self.mode}",
            )

            optimizer.step()

            self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
            self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            mask.data.clamp_(-1, 1)

            if self.args.run_eval_every_epoch:
                m = mask.cpu().detach().numpy().flatten()
                metrics["epoch"][f"epoch_{i}"] = {
                    'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'], data,
                                                           self.predict_fn, m,
                                                           kwargs['save_dir'], False)}

                if kwargs['save_perturbations']:
                    self.perturbation_manager.add_perturbation(
                        perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                        saliency=mask.cpu().detach().numpy(),
                        step=i, confidence=metrics['Confidence'][-1],
                        insertion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Insertion']['trap'],
                        deletion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Deletion']['trap'],
                        final_auc=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Final']['AUC'],
                        mean_gradient=metrics['Mean Gradient'][-1],
                        tv_norm=metrics['TV Norm'][-1],
                        main_loss=metrics["MSE"][-1],
                        budget=metrics["Budget"][-1],
                        total_loss=metrics["Total Loss"][-1],
                        category=metrics["Category"][-1],
                        saliency_sum=metrics["Saliency Sum"][-1],
                        saliency_var=metrics["Saliency Var"][-1])

            if self.enable_wandb:
                _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"},
                         **{"Gradient": wandb.Histogram(mask.grad.cpu().detach().numpy()),
                            "Training": [upsampled_mask, noise, perturbated_input,
                                         ] + [perturbation] if self.args.enable_blur else []
                            }
                         }
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]['eval_metrics']}

                wandb.log(_mets)

        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
            wandb.run.summary["pos_sum"] = np.sum(mask[np.argwhere(mask > 0)])
            wandb.run.summary["neg_sum"] = np.sum(mask[np.argwhere(mask < 0)])
        abs_mask = mask
        mask_min = np.min(abs_mask)
        mask_max = np.max(abs_mask)
        norm_mask = (mask - mask_min) / (mask_max - mask_min)
        save_timeseries(mask=norm_mask, raw_mask=mask, time_series=data.numpy(),
                        blurred=blurred_signal, save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb,
                        algo=self.args.algo, dataset=self.args.dataset, category=label)

        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)
            wandb.run.summary["norm_pos_sum"] = np.sum(norm_mask[np.argwhere(mask > 0)])
            wandb.run.summary["norm_neg_sum"] = np.sum(norm_mask[np.argwhere(mask < 0)])

        np.save("./r_mask", mask)
        np.save("./n_mask", norm_mask)
        return mask, self.perturbation_manager
