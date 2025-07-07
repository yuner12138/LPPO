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
# from PDTSPTest_Vector_select_best import PDTSPTest_Vector_select_best as Tester
from PDTSPTest_Vector_select import PDTSPTest_Vector_select as Tester
# from PDTSPTest_Vector import PDTSPTest_Vector as Tester
#from PDTSPTest_select_ben import PDTSPTest_select_ben as Tester
##########################################################################################
# parameters

env_params = {
    'problem_size': 101
}


model_params = {
    'max_unvisited':10,
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
    'n_iter':5,
    'problem_size':101
}

tester_params = {
    'number_k':128,
    'block_sample':1,
    'sample_select':False,
    'greedy_select_block':True,
    'greedy_action_selection_2':False,
    'usekmeans':True,
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'amp_inference': False,
    'model_load': {
        'path': './result_old/MLP_AVG_100_1100',  # directory path of pre-trained model and log files saved.
        'epoch': '1100',  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size1':1,
    'augmentation_enable': True,
    'aug_factor': 9,
    'aug_batch_size': 1,
    'test_z_sample_size': 500*25,
    'num_block':25,
    'num_vector_oneB':500,
    'EAS_params': {
        'enable': False,
        'iterations': 200,
        'lr': 0.0003,
        'lambda': 0.05,
        'resample': True
    },
    'test_data_load': {
        'enablepkl': True,
        'filenamepkl': './datasets/gaussian_100_pdtsp_1.pkl',
        'enablecvs': True,
        'filenamecvs': './pdtsp_benchmark/PDTSP201P9.csv',
        #'filename': '../datasets/pdp100_test_seed6666.pkl'
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
