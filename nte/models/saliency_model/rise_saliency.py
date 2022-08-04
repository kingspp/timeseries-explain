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


class RiseSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, num_masks: int, **kwargs):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.num_masks = num_masks
        self.softmax_fn = torch.nn.Softmax(dim=1)
        self.confidences = []
        self.perturbations = []

    def generate_saliency(self, data, label, **kwargs):
        MASKS = torch.randint(0, 2, (self.num_masks, data.shape[1]), dtype=torch.float)  # Random mask
        outer_accuracy = np.zeros((data.shape[1]))
        count_mask = np.zeros((data.shape[1]))
        for mask in MASKS:
            count_mask += mask.cpu().numpy()
            X_batch_masked = mask * data
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
