# -*- coding: utf-8 -*-
"""
| **@created on:** 8/29/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| Run Command :
| python3 main.py mse wafer dnn WaferDataset_model_dnn_loss_ce_lr_0.0005_epochs_200.ckpt  1
|
| **Sphinx Documentation Status:**
|
tasks = 4
samples_per_task = 3
data = list(range(12))

total_loops = 0

for TASK_ID in range(tasks):
    for e, s in enumerate(data[
                          int(TASK_ID) * samples_per_task: int(TASK_ID) * samples_per_task + samples_per_task]):
        cur_ind = e + (int(TASK_ID) * samples_per_task)
        print(f"Task: {TASK_ID}, Index: {cur_ind}, Data: {s}")
        total_loops+=1

print(f"Total Runs: {total_loops}")
|
|
|


"""
import torch
import numpy as np
import os
import json
import wandb
import ssl
from nte.experiment.utils import get_model, dataset_mapper, backgroud_data_configuration, get_run_configuration
import shortuuid
import matplotlib.pyplot as plt
from nte.models.saliency_model import SHAPSaliency, LimeSaliency
from nte.models.saliency_model.rise_saliency import RiseSaliency
from nte.models.saliency_model.mse_gradient_saliency import MSEGradientSaliency
from nte.models.saliency_model.nte_explainer import NTEGradientSaliency
from nte.models.saliency_model.random_saliency import RandomSaliency
from nte.models.saliency_model.nte_random_explainer import NTERandomSaliency
from nte.models.saliency_model.rise_w_replacement import RiseWReplacementSaliency
from nte.models.saliency_model.pert_explainer import PertSaliency
import random
from nte.experiment.evaluation import run_evaluation_metrics
from nte.experiment.default_args import parse_arguments
import seaborn as sns
from nte.experiment.utils import number_to_dataset, set_global_seed, replacement_sample_config
from nte.utils import CustomJsonEncoder

sns.set_style("darkgrid")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = True
WANDB_DRY_RUN = False

# ENABLE_WANDB = False
# WANDB_DRY_RUN = True

BASE_SAVE_DIR = 'results_v1/2312/'
BASE_SAVE_DIR = '/tmp/'
if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    args = parse_arguments()
    print("Config: \n", json.dumps(args.__dict__, indent=2))

    if args.dataset in number_to_dataset.keys():
        args.dataset = number_to_dataset[args.dataset]

    if args.enable_seed:
        set_global_seed(args.seed_value)

    ENABLE_SAVE_PERTURBATIONS = args.save_perturbations
    PROJECT_NAME = args.pname
    # PROJECT_NAME = args.eval_replacement + "_" + args.grad_replacement + "_" + args.dataset

    dataset = dataset_mapper(DATASET=args.dataset)

    TAG = f'{args.algo}-{args.dataset}-{args.background_data}-{args.background_data_perc}-run-{args.run_id}'
    BASE_SAVE_DIR = BASE_SAVE_DIR + "/" + TAG

    # todo Ramesh: Load black box model -> check this in utils.py
    model = get_model(dest_path=args.bbm_path, dataset=args.dataset,
                      use_cuda=use_cuda, bbm=args.bbm, multi_class=args.multi_class)
    softmax_fn = torch.nn.Softmax(dim=-1)

    bg_data, bg_len = backgroud_data_configuration(BACKGROUND_DATA=args.background_data,
                                                   BACKGROUND_DATA_PERC=args.background_data_perc,
                                                   dataset=dataset)

    print(
        f"Using {args.background_data_perc}% of background data. Samples: {bg_len}")

    config = args.__dict__

    if args.algo == 'pert':
        pert = PertSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
                            enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'nte-random':
        nte_rand = NTERandomSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                     predict_fn=model,
                                     enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'lime':
        lime = LimeSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model)
    elif args.algo == 'shap':
        shap = SHAPSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model)
    elif args.algo == 'rise':
        config["num_masks"] = 10000
        rise = RiseSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
                            num_masks=config['num_masks'])
    elif args.algo == 'random':
        random_s = RandomSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
                                  args=args, max_itr=10000)
    elif args.algo == 'rise-rep':
        config["num_masks"] = 10000
        rise_rep = RiseWReplacementSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                            predict_fn=model,
                                            args=args, num_masks=config['num_masks'])
    config = {**config, **{
        "tag": TAG,
        "algo": args.algo,
    }}

    dataset_len = len(dataset.test_data)

    ds = get_run_configuration(
        args=args, dataset=dataset, TASK_ID=args.task_id)

    for ind, (original_signal, original_label) in ds:
        try:
            if args.enable_seed_per_instance:
                set_global_seed(random.randint())
            metrics = {'epoch': {}}
            cur_ind = args.single_sample_id if args.run_mode == 'single' else (ind + (int(
                args.task_id) * args.samples_per_task))
            UUID = dataset.valid_name[cur_ind] if args.dataset_type == 'valid' else shortuuid.uuid(
            )
            EXPERIMENT_NAME = f'{args.algo}-{cur_ind}-R{args.run_id}-RT{args.r_index}-{UUID}'
            print(
                f" {args.algo}: Working on dataset: {args.dataset} index: {cur_ind} [{((cur_ind + 1) / dataset_len * 100):.2f}% Done]")
            SAVE_DIR = f'{BASE_SAVE_DIR}/{EXPERIMENT_NAME}'
            os.system(f'mkdir -p "{SAVE_DIR}"')
            os.system(f'mkdir -p "./wandb/{TAG}/"')
            config['save_dir'] = SAVE_DIR

            if args.run_mode == 'single' and args.dynamic_replacement == False:
                config = {**config, **replacement_sample_config(xt_index=args.single_sample_id, rt_index=args.r_index,
                                                                dataset=dataset, model=model,
                                                                dataset_type=args.background_data)}

            json.dump(config, open(SAVE_DIR + "/config.json", 'w'),
                      indent=2, cls=CustomJsonEncoder)
            if ENABLE_WANDB:
                wandb.init(entity="xai", project=PROJECT_NAME, name=EXPERIMENT_NAME,
                           tags=TAG, config=config, reinit=True, force=True, dir=f"./wandb/{TAG}/")

            plt.plot(original_signal, label="Original Signal")
            original_signal_norm = original_signal / np.max(original_signal)
            original_signal = torch.tensor(original_signal_norm, dtype=torch.float32)
            plt.xlabel("Timesteps")
            plt.ylabel("Values")

            with torch.no_grad():
                if args.bbm == 'rnn':
                    target = softmax_fn(model(original_signal.reshape(1, -1)))
                elif args.bbm == 'dnn':
                    target = softmax_fn(model(original_signal))
                else:
                    raise Exception(
                        f"Black Box model not supported: {args.bbm}")

            category = np.argmax(target.cpu().data.numpy())
            args.dataset = dataset
            if ENABLE_WANDB:
                wandb.run.summary[f"prediction_class"] = category
                wandb.run.summary[f"prediction_prob"] = np.max(
                    target.cpu().data.numpy())
                wandb.run.summary[f"label"] = original_label
                wandb.run.summary[f"target"] = target.cpu().data.numpy()

            if args.background_data == "none":
                    nte.background_data = original_signal
                    nte.background_label = original_label

            if args.algo == "pert":
                mask, perturbation_manager = pert.generate_saliency(
                    data=original_signal, label=original_label,
                    save_dir=SAVE_DIR, save_perturbations=args.save_perturbations, target=target, dataset=dataset)

            elif args.algo == "nte-random":
                mask, perturbation_manager = nte_rand.generate_saliency(
                    data=original_signal, label=original_label,
                    save_dir=SAVE_DIR, save_perturbations=args.save_perturbations, target=target, dataset=dataset)

            elif args.algo == "cm":
                mask, perturbation_manager = cm.generate_saliency(
                    data=original_signal, label=original_label,
                    save_dir=SAVE_DIR, save_perturbations=args.save_perturbations, target=target, dataset=dataset)

            elif args.algo == 'lime':
                mask, perturbation_manager = lime.generate_saliency(
                    data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                    save_perturbations=args.save_perturbations, target=target, dataset=dataset, save_dir=SAVE_DIR, )

            elif args.algo == 'shap':
                mask, perturbation_manager = shap.generate_saliency(
                    data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                    save_perturbations=args.save_perturbations, target=target, dataset=dataset, save_dir=SAVE_DIR, )

            elif args.algo == 'rise':
                mask, perturbation_manager = rise.generate_saliency(
                    data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                    save_perturbations=args.save_perturbations, target=target, dataset=dataset, save_dir=SAVE_DIR, )

            elif args.algo == 'random':
                mask, perturbation_manager = random_s.generate_saliency(
                    data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                    save_perturbations=args.save_perturbations, target=target, dataset=dataset, save_dir=SAVE_DIR, )

            elif args.algo == 'rise-rep':
                mask, perturbation_manager = rise_rep.generate_saliency(
                    data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                    save_dir=SAVE_DIR, save_perturbations=args.save_perturbations, target=target, dataset=dataset)
            mask = mask.flatten()

            # Evaluation Metrics
            if args.run_eval:
                metrics['eval_metrics'] = run_evaluation_metrics(args.eval_replacement, dataset, original_signal, model,
                                                                 mask, SAVE_DIR, ENABLE_WANDB, main_args=args)
                if ENABLE_WANDB:
                    for mk, mv in metrics['eval_metrics'].items():
                        for k, v in mv.items():
                            if isinstance(v, list):
                                pass
                            else:
                                wandb.run.summary[f"{mk} {k}"] = v
                    wandb.run.summary["Saliency Sum Final"] = np.sum(mask)
                    wandb.run.summary["Saliency Var Final"] = np.var(mask)

                json.dump(metrics, open(
                    SAVE_DIR + "/metrics.json", 'w'), indent=2)

            if args.save_perturbations:
                perturbation_manager.to_csv(
                    SAVE_DIR=SAVE_DIR, TAG=TAG, UUID=UUID, SAMPLE_ID=cur_ind)
                if ENABLE_WANDB:
                    wandb.save(
                        f"{SAVE_DIR}/perturbations-{TAG}-{UUID}-{cur_ind}.csv")
        except Exception as e:
            with open(f'logs/{TAG}_error.log', 'a+') as f:
                f.write(cur_ind)
                f.write(e.__str__())
                f.write("\n\n")
