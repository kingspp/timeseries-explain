# -*- coding: utf-8 -*-
"""
| **@created on:** 9/22/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

__all__ = ['parse_arguments']

import argparse
from nte.experiment.utils import str2bool


def parse_arguments(standalone=False):

    parser = argparse.ArgumentParser(description='NTE Pipeline')

    # General Configuration
    parser.add_argument('--pname', type=str, help='Project name - [project_name]', default="untitled")
    parser.add_argument('--task_id', type=int, help='Task ID', default=0)
    parser.add_argument('--run_id', type=str, help='Run ID', default=0)
    parser.add_argument('--save_perturbations', type=str2bool, nargs='?', const=False, help='Save Perturbations',
                        default=False)
    parser.add_argument('--conf_thres', type=float, help="Confidence threshold of prediction", default=0.0)

    # Run Configuration
    parser.add_argument('--run_mode', type=str, help='Run Mode - ["single", "local", "turing"]',
                        default='single', choices=["single", "local", "turing"])
    parser.add_argument('--dataset_type', type=str, help='Run Mode - ["train", "test", "valid"]',
                        default='test', choices=["train", "test", "valid"])
    parser.add_argument('--samples_per_task', type=int, help='Number of samples to run per task in turing mode',
                        default=10)
    parser.add_argument('--jobs_per_task', type=int, help='Max number of jobs to run all samples',
                        default=-1)
    parser.add_argument('--single_sample_id', type=int, help='Single Sample',
                        default=84) #24

    # Seed Configuration
    parser.add_argument('--enable_seed', type=str2bool, nargs='?', const=True, help='Enable Seed',
                        default=True)
    parser.add_argument('--enable_seed_per_instance', type=str2bool, nargs='?', const=True,
                        help='Enable Seed Per Instance',
                        default=False)
    parser.add_argument('--seed_value', type=int, help='Seed Value',
                        default=0)

    # Mask Normalization
    parser.add_argument('--mask_norm', type=str, help='Mask Normalization - ["clamp", "softmax", "sigmoid"]',
                        default='clamp',
                        choices=["clamp", "softmax", "sigmoid", "none"])

    # Algorithm
    parser.add_argument('--algo', type=str, help='Algorithm type required - [mse|cm|nte|lime|shap|rise]',
                        default='lime',
                        choices=["mse", "cm", "lime", "nte", "shap", "rise", "random", "rise-rep", "p-nte", "p-nte-v1",
                                 "p-nte-v2", "p-nte-v3",  "p-nte-v4", "p-nte-v5", "cmwr", "nte-kl", "nte-dual", "ts-pert"])
    parser.add_argument('--grad_replacement', type=str, help='Gradient Based technique replacement strategy',
                        # default='random_opposing_instance',
                        default='random_instance',
                        choices=["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance"])
    parser.add_argument('--dynamic_replacement', type=str2bool, nargs='?', const=True,
                        help='Dynamic pick of Zt on every epoch', default=True)
    parser.add_argument('--r_index', type=int, help='Replacement Index',
                        default=-1)
                        # default=139)

    # Dataset and Background Dataset configuration
    parser.add_argument('--dataset', type=str, default='1',
                        help='Dataset name required - [blip|wafer|gun_point|ford_a|ford_b|earthquakes|ptb|ecg]',
                        choices=["blip", "wafer", "cricket_x", "gun_point", "earthquakes", "computers",
                                 "ford_a", "ford_b", "ptb","ecg","1", "2", "3", "4", "5", "6", "7","8","9"])
    parser.add_argument('--background_data', type=str, help='[train|test|none]', default='test',
                        choices=["train", "test", "none"])
    parser.add_argument('--background_data_perc', type=float, help='%% of Background Dataset', default=100)

    # Black-box model configuration
    parser.add_argument('--bbm', type=str, help='Black box model type - [dnn|rnn]', default='dnn',
                        choices=["dnn", "rnn"])
    parser.add_argument('--bbm_path', type=str, help='Black box model path - [dnn|rnn]', default="default")

    # Gradient Based Algo configurations
    parser.add_argument('--enable_blur', type=str2bool, nargs='?', const=True, help='Enable blur', default=False)
    parser.add_argument('--enable_tvnorm', type=str2bool, nargs='?', const=True, help='Enable TV Norm', default=True)
    parser.add_argument('--enable_budget', type=str2bool, nargs='?', const=True, help='Enable budget', default=True)
    parser.add_argument('--enable_noise', type=str2bool, nargs='?', const=True, help='Enable Noise', default=True)
    parser.add_argument('--enable_dist', type=str2bool, nargs='?', const=True, help='Enable Dist Loss', default=False)
    # parser.add_argument('--enable_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
    # parser.add_argument('--enable_weighted_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
    # parser.add_argument('--enable_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable EUC Loss',
    #                     default=False)
    # parser.add_argument('--enable_weighted_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable W EUC Loss',
    #                     default=False)
    parser.add_argument('--enable_lr_decay', type=str2bool, nargs='?', const=True, help='LR Decay', default=False)

    parser.add_argument('--dist_loss', type=str, help='Distance Loss Type - ["euc", "dtw", "w_euc", "w_dtw"]',
                        default='euc',
                        choices=["euc", "dtw", "w_euc", "w_dtw", "n_dtw", "n_w_dtw", "no_dist"])

    # Early Stopping Criteria
    parser.add_argument('--early_stopping', type=str2bool, nargs='?', const=False, help='Early Stop',
                        default=False)
    parser.add_argument('--early_stop_criteria_patience', type=float, help='Early Stop Criteria Patience',
                        default=100)
    parser.add_argument('--early_stop_criteria_perc', type=float, help='Early Stop Criteria Percentage',
                        default=10)

    # Evaluation Metric Configuration
    parser.add_argument('--run_eval', type=str2bool, nargs='?', const=True, help='Run Evaluation Metrics',
                        default=True)
    parser.add_argument('--run_eval_every_epoch', type=str2bool, nargs='?', const=True,
                        help='Run Evaluation Metrics for every epoch',
                        default=False)
    parser.add_argument('--eval_replacement', type=str,
                        help='Replacement Timeseries for evaluation [zeros|class_mean|instance_mean]',
                        default='class_mean',
                        choices=["zeros", "class_mean", "instance_mean"])

    # Hyper Param Configuration
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001) #0.01
    parser.add_argument('--lr_decay', type=float, help='LR Decay', default=0.999)
    parser.add_argument('--l1_coeff', type=float, help='L1 Coefficient', default=1) #0.05
    parser.add_argument('--tv_coeff', type=float, help='TV Norm Coefficient', default=0.1)
    parser.add_argument('--tv_beta', type=float, help='TV Norm Beta', default=3)
    parser.add_argument('--dist_coeff', type=float, help='Dist Loss Coeff', default=0.0)
    parser.add_argument('--w_decay', type=float, help='Weight Decay', default=0.0)
    # parser.add_argument('--dtw_coeff', type=float, help='Soft DTW Coeff', default=1)
    # parser.add_argument('--euc_coeff', type=float, help='EUC Loss Coeff', default=1)
    parser.add_argument('--mse_coeff', type=float, help='MSE Coefficient', default=2)
    parser.add_argument('--max_itr', type=int, help='Maximum Iterations', default=5)

    parser.add_argument('--bwin', type=int, help='bwin ', default=1)
    parser.add_argument('--run', type=str, help='Run ', default=1 )
    parser.add_argument('--bsigma', type=float, help='bsigma ', default=0.9)
    parser.add_argument('--sample', type=float, help='sample', default=6.25)

    if standalone:
        return parser.parse_known_args()
    else:
        return parser.parse_args()
