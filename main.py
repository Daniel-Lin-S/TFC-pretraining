"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch

from TFC.utils import _logger
from TFC.model import TFC, target_classifier
from TFC.dataloader import data_generator
from TFC.trainer import Trainer


# Args selections
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Description of current run, used in saving the results.')
parser.add_argument('--seed', default=42, type=int,
                    help='Random seed for reproducibility.')

parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    choices=['pre_train', 'fine_tune_test'],
                    help='pre_train means self-supervised pre-training; '
                    'fine_tune_test means supervised fine-tuning and testing.')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    choices=['SleepEEG', 'FD_A', 'HAR', 'ECG'],
                    help='The dataset to be used for pre-training: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    choices=['Epilepsy', 'FD_B', 'Gesture', 'EMG'],
                    help='The dataset to be used for fine-tuning: SleepEEG, FD_A, HAR, ECG')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata)

training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

module_name = f"config_files.{pretrain_dataset}_Configs"
configs_module = importlib.import_module(module_name)
Configs = getattr(configs_module, "Config")
configs = Configs()

####### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(
    logs_save_dir, experiment_description, run_description,
    training_mode + f"_seed_{SEED}_2layertransformer"
)
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  TF-C')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"datasets/{pretrain_dataset}"
targetdata_path = f"datasets/{targetdata}"

if not os.path.exists(sourcedata_path):
    raise ValueError(f"Cannot find sourcedata_path {sourcedata_path}")
if not os.path.exists(targetdata_path):
    raise ValueError(f"Cannot find targetdata_path {targetdata_path}")

subset = True  # if subset= true, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(
    sourcedata_path, targetdata_path, configs, training_mode, subset = subset)
logger.debug(
    "Loaded {}, {}, {} samples for training, validation, testing.".format(
        len(train_dl), len(valid_dl), len(test_dl))
)

# Load Model
TFC_model = TFC(configs).to(device)  # feature extractor
classifier = target_classifier(configs).to(device)

model_optimizer = torch.optim.Adam(
    TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(
    classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

start_time = datetime.now()

# Trainer
Trainer(TFC_model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode)

logger.debug(f"Total Training time : {datetime.now()-start_time}")
