# -*- coding: utf-8 -*-
"""
| **@created on:** 9/19/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""
from nte.models.saliency_model import Saliency
import numpy as np
import torch
from nte.utils.perturbation_manager import PerturbationManager
import random


class RiseWReplacementSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, num_masks: int, args=None):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.num_masks = num_masks
        self.softmax_fn = torch.nn.Softmax(dim=1)
        self.confidences = []
        self.perturbations = []
        self.args = args

    def dynamic_pick_zt(self, kwargs, data, label):
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
                Zt = torch.tensor(self.background_data[random.randrange(0, len(self.background_data))],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    sds = kwargs['dataset'].test_class_0_indices
                    sls = len(sds)
                else:
                    sds = kwargs['dataset'].test_class_1_indices
                    sls = len(sds)
                Zt = torch.tensor(sds[random.randrange(0, sls)], dtype=torch.float32)

            # a = original_signal.cpu().detach().numpy().flatten()
            # b = upsampled_mask.cpu().detach().numpy().flatten()
            # diff = np.array([np.linalg.norm(x - y) for x, y in zip(a, b)])
            # diff_norm = (diff - diff.min()) / (diff.max() - diff.min()) if np.sum(diff) > 0 else diff
            # p1 = original_signal.mul(upsampled_mask)
            # p2 = torch.tensor(diff_norm, dtype=torch.float32).mul(Mx)
            # perturbated_input = p1 + p2
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
                Zt = torch.tensor(self.background_data[random.randrange(0, len(self.background_data))],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:

                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][0],
                                      dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][1],
                                      dtype=torch.float32)

            # a = original_signal.cpu().detach().numpy().flatten()
            # b = upsampled_mask.cpu().detach().numpy().flatten()
            # diff = np.array([np.linalg.norm(x - y) for x, y in zip(a, b)])
            # diff_norm = (diff - diff.min()) / (diff.max() - diff.min()) if np.sum(diff) > 0 else diff
            # p1 = original_signal.mul(upsampled_mask)
            # p2 = torch.tensor(diff_norm, dtype=torch.float32).mul(Mx)
            # perturbated_input = p1 + p2
        return Zt

    def generate_saliency(self, data, label, **kwargs):
        MASKS = torch.randint(0, 2, (self.num_masks, data.shape[1]), dtype=torch.float)  # Random mask
        outer_accuracy = np.zeros((data.shape[1]))
        count_mask = np.zeros((data.shape[1]))

        for mask in MASKS:
            count_mask += mask.cpu().numpy()
            Zt = self.dynamic_pick_zt(kwargs=kwargs, data=data, label=label)
            X_batch_masked = mask * data + Zt.mul(1 - mask)
            self.perturbations.append(X_batch_masked.cpu().detach().numpy())
            # X_batch_masked = torch.reshape(torch.tensor(X_batch_masked, dtype=torch.float32), (152, 1, 1))
            res = self.softmax_fn(self.predict_fn(X_batch_masked))
            predictions = torch.argmax(res).item()
            self.confidences.append(np.max(res.cpu().detach().numpy()))
            outer_accuracy += (1 * mask if predictions == label else 0 * mask).cpu().detach().numpy()
        saliency = outer_accuracy / count_mask
        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(self.perturbations,
                                                          confidences=self.confidences)
        return saliency, self.perturbation_manager
