##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# Pytorch Config

import torch
#torch.backends.cuda.matmul.allow_tf32 = False  # see https://dev-discuss.pytorch.org/t/pytorch-and-tensorfloat32/504

# Attention implementation. PyTorch picks one of the enabled methods. If you want to force one, you need to disable all others.
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src, get_result_folder
from PDTSPTrainer import PDTSPTrainer as Trainer
#from PDTSPCollect import PDTSPCollect as Collect

##########################################################################################
# parameters

run_params = {
    'name': None
}


env_params = {
    'problem_size': 101,
    'customer_size': 50,
    'rollout_size': 128,
    'capacity': 20,
    'demand_min': 1,
    'demand_max': 10,
}

model_params = {
    'max_unvisited': 10,
    'embedding_dim': 128,
    'lppo_embedding_dim': 512,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'z_dim': 16,
    'use_fast_attention': False,  # Use PyTorch implementation for attention
    'force_first_move': False,
    'em_clusters':5,
    'n_iter':5,
    'problem_size': 101
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [1000,],
        'gamma': 0.1
    }
}


collect_params={

}


trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'amp_training': False,
    'compile_model': False,  # Does not work correctly at the moment
    'epochs': 500,
    'train_num_rollouts': 128*100000,
    'train_batch_size': 50,
    'train_z_sample_size': 128,
    'mask_leak_alpha': 0,  # Not needed if fist move is not forced
    'val_episodes': 10,
    'val_batch_size': 10,
    'val_z_sample_size': 8,
    'logging': {
        'model_save_interval': 10,
    },
    'model_load': {
        'pomo_enable': False,
        'enable': False,  # enable loading pre-trained model
        'path': './result/mpdtsp100',  # directory path of pre-trained model and log files saved.
        'epoch': '800',  # epoch version of pre-trained model to laod.
    },
    'validation_data_load': {
        'enable': False,
        'filename': '../mdatasets/uniform100_pdtsp.pkl'
    },
    'augmentation': {
        'aug_8': True,
        'aug_9': False,
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_PDTSP_n100_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params, run_name=run_params['name'])
    _print_config()

    trainer = Trainer(run_params=run_params,
                      env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_solutions'] = 500
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
