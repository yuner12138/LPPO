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
# torch.backends.cuda.matmul.allow_tf32 = True  # see https://dev-discuss.pytorch.org/t/pytorch-and-tensorfloat32/504
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
from PDTSPTester import PDTSPTester as Tester

##########################################################################################
# parameters

env_params = {
    'problem_size': 51,
    'customer_size': 25,
    'rollout_size': 125,
    'capacity': 20,
    'demand_min': 1,
    'demand_max': 10,
    'load_path': '',
    'synthetic_dataset': True,
    'sample': False,
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
    'eval_type': 'greedy1',
    'z_dim': 16,
    'use_fast_attention': False,
    'force_first_move': False,
    'em_clusters':5,
    'n_iter':5,
    'problem_size':env_params['problem_size'],
    'sample':  True if env_params['sample'] else False,
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'amp_inference': False,
    'model_load': {
        'path': './result/mpdtsp100',  # directory path of pre-trained model and log files saved.
        'epoch': '800',  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 100,
    'test_batch_size1': 10,
    'augmentation_enable': True,
    'aug_factor': 9,
    'aug_batch_size': 10,
    'test_z_sample_size': env_params['rollout_size'],
    'EAS_params': {
        'enable': False,
        'iterations': 200,
        'lr': 0.0003,
        'lambda': 0.05,
        'resample': True,
    },
    'sample':  True if env_params['sample'] else False,
    'sample_batch': 1,
    'sample_size': 64,
    'test_data_load': {
        'enable': True,
        'filename': './mdatasets/uniform100_pdtsp.pkl'
    },
    'solution_max_length': 300,  # for buffer length storing solution
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_pdtsp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
