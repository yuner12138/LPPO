import numpy as np
import torch
from logging import getLogger

from PDTSPEnv import PDTSPEnv as Env
#from PDTSPModel import PDTSPModel as Model
#from PDTSPModel_cluster import PDTSPModel_cluster as Model
#from PDTSPModel_unvisted import PDTSPModel_unvisted as Model
#from PDTSPModel_POMO import PDTSPModel_POMO as Model
#from PDTSPModel_unvisted_avg import PDTSPModel_unvisted_avg as Model
#from PDTSPModel_unvisted_AVG_nop import PDTSPModel_unvisted_AVG_nop as Model
from PDTSPModel_unvisted_MLP_padavg import PDTSPModel_unvisted_MLP_padavg as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
import itertools


class PDTSPTrainer:
    def __init__(self,
                 run_params,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result_old folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(self.model_params,
                           self.trainer_params)
        if self.trainer_params['compile_model']:
            self.model = torch.compile(self.model)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler()

        # Restore
        self.start_epoch = 1
        if run_params["name"] is not None:
            # try to resume named training run
            model_load_path = os.path.join("result_old", "run_" + run_params["name"], "checkpoint-100-cluster.pt")
            print(model_load_path)
            if os.path.exists(model_load_path):
                checkpoint = torch.load(model_load_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                self.start_epoch = 1 + checkpoint['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scheduler.last_epoch = self.start_epoch - 1
                print('faw')
                self.logger.info('Resuming named training run!')
        if trainer_params['model_load']['pomo_enable'] and self.start_epoch == 1:
            ACTOR_fullname = '{path}/ACTOR_state_dic.pt'.format(
                path='result_old/20240407_1439__pdp20_only_heterogeneous_encoder3/CheckPoint_ep00081')
            LRSTER_fullname = '{path}/LRSTEP_state_dic.pt'.format(
                path='result_old/20240407_1439__pdp20_only_heterogeneous_encoder3/CheckPoint_ep00081')
            OPTIM_fullname = '{path}/OPTIM_state_dic.pt'.format(
                path='result_old/20240407_1439__pdp20_only_heterogeneous_encoder3/CheckPoint_ep00081')
            ACTOR_full = torch.load(ACTOR_fullname, map_location=device)
            LRSTER_full = torch.load(LRSTER_fullname, map_location=device)
            OPTIM_full = torch.load(OPTIM_fullname, map_location=device)

            # pretrained_state_dict = pretrained_model.state_dict()
            # new_state_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
            # self.model.load_state_dict(new_state_dict, strict=False)
            self.model.load_state_dict(ACTOR_full, strict=False)
            # self.optimizer.load_state_dict(OPTIM_full)
            # self.scheduler.load_state_dict(LRSTER_full)
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'],strict=False)
            print('Saved Pomo-Model Loaded !!')

        if trainer_params['model_load']['enable'] and self.start_epoch == 1:
            # start new training run from POMO base model or continue Lppo training
            model_load_path = '{path}/checkpoint-{epoch}.pt'.format(**trainer_params['model_load'])
            checkpoint = torch.load(model_load_path, map_location=device)
            if "decoder.lppo_layer_1.weight" in checkpoint['model_state_dict'].keys():
                # Loaded model is Lppo model
                # self.start_epoch = 1 + checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.start_epoch = 1 + checkpoint['epoch']
            self.logger.info('Saved Model Loaded!')
        if self.trainer_params['compile_model']:
            self.model = torch.compile(self.model)
        self.env = Env(**self.env_params)

        #只更新部分参数的代码
        # submodule_params = [p for p in self.model.collect.parameters() if p.requires_grad]
        # self.optimizer = Optimizer(submodule_params, **self.optimizer_params['optimizer'])
        # #self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        # self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        # self.scaler = torch.cuda.amp.GradScaler()
        # utility
        self.time_estimator = TimeEstimator()
        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])

    def run(self):
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            # print("xuexilv ", self.optimizer.param_groups[0]['lr'])
            self.logger.info('=================================================================')
            # Train
            print("xuexilv ", self.optimizer.param_groups[0]['lr'])
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay
            self.scheduler.step()
            # Validate
            # self.logger.info("Starting validation")
            # self.logger.info("Greedy:")
            # self._validate(greedy_construction=True)
            # self.logger.info("Sampling:")
            # self._validate(greedy_construction=False)
            # self.logger.info("Sampling Aug:")
            # self._validate(greedy_construction=False, use_augmentation=True)


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            self.logger.info("Saving trained_model")
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'result_log': self.result_log.get_raw_data(),
                'z_dim': self.model_params['z_dim'],
                'force_first_move': self.model_params["force_first_move"]
            }
            #torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
            if all_done or (epoch % model_save_interval) == 0:
                print(self.result_folder)
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        if self.model_params['force_first_move']:
            solutions_per_instance = self.trainer_params["train_z_sample_size"] * self.env_params['problem_size']
        else:
            solutions_per_instance = self.trainer_params["train_z_sample_size"]

        # Number of batches per epoch
        train_num_episode = int(self.trainer_params['train_num_rollouts'] / solutions_per_instance)

        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt >= 1:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        z_sample_size = self.trainer_params['train_z_sample_size']
        z_dim = self.model_params['z_dim']
        amp_training = self.trainer_params['amp_training']
        device = "cuda" if self.trainer_params['use_cuda'] else "cpu"

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size


        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size, rollout_size)

        reset_state, group_state, _, _= self.env.reset()
        #@#
        group_s = reset_state.problems.size(1) // 2

        # Sample z vectors
        z = self.sample_z_vectors(batch_size, starting_points, z_dim, z_sample_size, rollout_size)

        self.model.pre_forward(reset_state, z)
        # self.model.pre_forward_C(reset_state, self.binary_string_pool)
        # # print("进入collect")
        # z = self.model.collect(z_sample_size)
        # self.model.decoder.set_z(z)

        #self.model.pre_forward(reset_state, z)

        prob_list = torch.zeros(size=(self.env.batch_size, self.env.rollout_size, 0))
        # shape: (batch, rollout, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        #@#
        #prob = torch.ones(size=(batch_size, self.env.Bit_size))
        first_selected = torch.zeros(size=(batch_size, self.env.rollout_size))
        state, reward, done = self.env.step(first_selected)

        while not done:

            selected, prob = self.model(state)
            # shape: (batch, rollout)
            selected_clone = selected.clone()
            selected_clone[group_state.finished] = 0
            state, reward, done = self.env.step(selected_clone)

            prob_clone = prob.clone()
            prob_clone[group_state.finished] = 1
            prob_list = torch.cat((prob_list, prob_clone[:, :, None]), dim=2)

        # POMO Loss
        reward_pop = reward.reshape(batch_size, z_sample_size, -1)
        if self.model_params['force_first_move']:
            # Mean is calculated over different POMO rollouts
            advantage = reward_pop - reward_pop.mean(dim=2, keepdim=True)
        else:
            # Mean is calculated over different z samples
            advantage = reward_pop - reward_pop.mean(dim=1, keepdim=True)
        advantage = advantage.reshape(batch_size, -1)
        # shape: (batch, rollout)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, rollout)

        # Finding the best rollout of each z sample
        costs = -reward.reshape(batch_size, z_sample_size, -1)
        best_idx = costs.argsort(1).argsort(1)
        best_idx = best_idx.reshape(batch_size, -1)
        mask = best_idx < 1
        mask = torch.clamp(mask + (self.trainer_params["mask_leak_alpha"] / z_sample_size), max=1)
        log_prob *= mask

        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, rollout)
        loss_mean = loss.mean()
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()

        if not amp_training:

            loss_mean.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss_mean).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        #print(self.model.decoder.lppo_layer_1.weight.datasets)
        # print("score_mean",score_mean)
        return score_mean.item(), loss_mean.item()

    def _validate(self, greedy_construction=False, use_augmentation=False):
        val_num_episode = self.trainer_params['val_episodes']
        z_sample_size = self.trainer_params['val_z_sample_size']
        z_dim = self.model_params['z_dim']
        batch_size = self.trainer_params['val_batch_size']

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size

        if use_augmentation:
            aug_factor = 8
            batch_size = max(1, batch_size // aug_factor)
        else:
            aug_factor = 1

        self.model.eval()

        if self.trainer_params['validation_data_load']['enable']:
            self.env.use_pkl_saved_problems(self.trainer_params['validation_data_load']['filename'], val_num_episode)

        costs = torch.zeros(size=(0, starting_points, z_sample_size * aug_factor))
        mean_log_prob = []

        episode = 0
        while episode < val_num_episode:

            remaining = val_num_episode - episode
            batch_size = min(batch_size, remaining)
            self.env.load_problems(batch_size, rollout_size, aug_factor)
            episode += batch_size

            with torch.no_grad():
                reset_state, _, _ = self.env.reset()

                # Sample z vectors
                z = self.sample_z_vectors(batch_size * aug_factor, starting_points, z_dim, z_sample_size, rollout_size)
                self.model.pre_forward(reset_state, z)

                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                prob_list = torch.zeros(size=(batch_size * aug_factor, self.env.rollout_size, 0))
                while not done:
                    selected, prob = self.model(state, greedy_construction)
                    # shape: (batch, rollout)
                    state, reward, done = self.env.step(selected)
                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

                reward = reward.reshape(aug_factor, batch_size, z_sample_size, starting_points).transpose(0, 1)
                reward = reward.reshape(batch_size, aug_factor*z_sample_size, starting_points).transpose(1, 2)
                costs = torch.cat((costs, -reward), dim=0)
                mean_log_prob.append(prob_list.log().sum(2).mean().item())

        num_unique_costs = torch.zeros(size=(val_num_episode, starting_points))

        for i in range(num_unique_costs.size(0)):
            for j in range(num_unique_costs.size(1)):
                num_unique_costs[i][j] = torch.unique(costs[i, j]).numel()

        mean_unique = num_unique_costs.mean()/(aug_factor*z_sample_size)
        cost_best = costs.min(dim=2)[0].mean()
        cost_pomo = costs.min(dim=2)[0].min(dim=1)[0].mean()
        self.logger.info(
            f'Log prob: {np.array(mean_log_prob).mean():.4f} Percentage of unique costs: {mean_unique:.3f} Costs (mean, best, best pomo): {costs.mean():.4f} {cost_best:.4f} {cost_pomo:.4f}')

        self.env.use_random_problems()  # Clear validation problem datasets from env

    def sample_z_vectors(self, batch_size, starting_points, z_dim, z_sample_size, rollout_size):

        if 2**z_dim == rollout_size:
            z = self.binary_string_pool[None].expand(batch_size, rollout_size, z_dim)
        else:
            z_idx = torch.multinomial((torch.ones(batch_size * starting_points, 2**z_dim) / 2**z_dim),
                                  z_sample_size, replacement=z_sample_size > 2**z_dim)
            z = self.binary_string_pool[z_idx].reshape(batch_size, starting_points, z_sample_size, z_dim)
            z = z.transpose(1, 2).reshape(batch_size, rollout_size, z_dim)
        return z
