import random
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_random_state
import numpy as np
import os
from logging import getLogger
# from julei import *
# from PDTSPEnv import PDTSPEnv as Env
from PDTSPEvalEnv import PDTSPEvalEnv as Env
#from PDTSPModel import PDTSPModel as Model
#from PDTSPModel import PDTSPModel as Model
#from PDTSPModel_cluster import PDTSPModel_cluster as Model
#from PDTSPModel_unvisted import PDTSPModel_unvisted as Model
#from PDTSPModel_POMO import PDTSPModel_POMO as Model
from PDTSPModel_unvisted_MLP_padavg import PDTSPModel_unvisted_MLP_padavg as Model
from utils.utils import *
import itertools
from torch.optim import Adam as Optimizer

class PDTSPTest_Vector_select:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.begin_idx = 0
        # result_old folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(self.model_params,
                           self.tester_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        four_digit_number = random.randint(1000, 9999)
        seed = four_digit_number
        print("seed={}".format(four_digit_number))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        self.binary_string_pool = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=model_params['z_dim'])])

    def run(self):
        self.k = self.tester_params['number_k']
        self.X = self.binary_string_pool.to("cpu").numpy()
        # 初始化KMeans对象并拟合数据
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(self.X)
        # 获取聚类中心和标签
        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_

        # print(len(self.X))
        # clusters, centroids = bisecting_kmeans(X=self.X, K=100)
        # self.centroids = clusters
        # self.labels = centroids

        # 找出每个聚类中离聚类中心最近的向量
        closest_vectors = []
        for i in range(self.k):
            # 获取属于第i个聚类的所有点
            cluster_points = self.X[self.labels == i]
            # 计算这些点到聚类中心的距离
            distances = np.linalg.norm(cluster_points - self.centroids[i], axis=1)
            # 找到距离最小的点的索引
            closest_idx = np.argmin(distances)
            # 添加这个点到closest_vectors列表中
            closest_vectors.append(cluster_points[closest_idx])

        # 将列表转换为numpy数组（如果需要的话）
        self.closest_vectors = np.array(closest_vectors)
        self.closest_vectors = torch.from_numpy(self.closest_vectors)
        print('聚类完成')
        if not self.tester_params['greedy_select_block']:
            self.block_sample = self.tester_params['block_sample']
            self.z_vectors_block = torch.zeros((self.k, self.block_sample, 16))
            for i in range(self.k):
                cluster_points = self.X[self.labels == i]
                # 随机采样
                z_size = torch.from_numpy(cluster_points).size(dim=0)
                z_idx = torch.multinomial(
                    (torch.ones(1, z_size) / z_size),
                    self.block_sample, replacement=self.block_sample > z_size).cpu()
                self.z_vectors_block[i] = torch.from_numpy(cluster_points[z_idx])
        #print("z vectors block size",self.z_vectors_block.size())
        #print(self.z_vectors_block)

        # if self.tester_params['greedy_select_block']:
        #     self.z_vectors_block

        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']


        if self.tester_params['test_data_load']['enable']:
            self.data_file = self.tester_params['test_data_load']['filename']

        episode = 0

        while episode < test_num_episode:
            # print(episode)
            remaining = test_num_episode - episode

            batch_size = min(self.tester_params['test_batch_size1'], remaining)


            if not self.tester_params['EAS_params']['enable']:
                score, aug_score = self._test_one_batch(batch_size)
            else:
                score, aug_score = self._search_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)


            if all_done:

                print(" *** Test Done *** ")
                print(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                print(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):
        num_block = self.tester_params['num_block']
        num_vector_oneB = self.tester_params['num_vector_oneB']
        z_sample_size = self.tester_params['test_z_sample_size']
        z_dim = self.model_params['z_dim']
        amp_inference = self.tester_params['amp_inference']
        device = "cuda" if self.tester_params['use_cuda'] else "cpu"
        greedy_action_selection = self.model_params['eval_type'] == 'greedy'

        if self.model_params['force_first_move']:
            starting_points = self.env_params['problem_size']
            rollout_size = starting_points * z_sample_size
        else:
            starting_points = 1
            rollout_size = z_sample_size

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.rollout = self.env.rollout_size
            self.env.rollout_size = self.k
            if self.tester_params['greedy_select_block']:
                self.env.get_dataset_problem_OURS(self.data_file,batch_size,self.begin_idx,self.env_params['capacity'],aug_factor)
            else:
                self.env.get_dataset_problem_OURS(self.data_file,batch_size,self.begin_idx,self.env_params['capacity'],aug_factor)

            reset_state, _, _ = self.env.reset()

            if self.tester_params['usekmeans']:
                if self.tester_params['greedy_select_block']:
                    # # 将NumPy数组转换为PyTorch张量 (原代码)
                    z = self.closest_vectors
                    z = z.unsqueeze(0).repeat(batch_size * aug_factor, 1,1).reshape(batch_size * aug_factor, starting_points, self.k, z_dim)
                    z = z.to(device)
                    #print("z size",z.size())

                else:
                    #一个块sample10个
                    z = self.z_vectors_block.reshape(-1,z_dim)

                    #print("z size",z.size())
                    #print(z)
                    z = z.unsqueeze(0).repeat(batch_size * aug_factor, 1,1).reshape(batch_size * aug_factor, starting_points, self.block_sample*self.k, z_dim)
                    z = z.to(device)


            else:
                z_idx = torch.multinomial(
                    (torch.ones(batch_size * aug_factor * starting_points, 2 ** z_dim) / 2 ** z_dim),
                                           z_sample_size, replacement=z_sample_size > 2**z_dim)
                z = self.binary_string_pool[z_idx].reshape(batch_size * aug_factor, starting_points, z_sample_size, z_dim)

            if self.tester_params['augmentation_enable']:
                if self.tester_params['greedy_select_block']:
                    z = z.transpose(1, 2).reshape(batch_size * aug_factor, self.k, z_dim)
                else:
                    z = z.transpose(1, 2).reshape(batch_size * aug_factor, self.k * self.block_sample, z_dim)
                    #print("z ",z)
            else:
                #z = z.reshape(batch_size * aug_factor, self.k*self.block_sample, z_dim)
                z = z.reshape(batch_size * aug_factor, self.k , z_dim)

            self.model.pre_forward(reset_state, z)
            prob_list = torch.zeros(size=(self.env.batch_size, self.env.rollout_size, 0))
            # shape: (batch, rollout, 0~problem)
            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()

            # @#
            # prob = torch.ones(size=(batch_size, self.env.Bit_size))
            first_selected = torch.zeros(size=(batch_size*aug_factor, self.env.rollout_size))
            state, reward, done = self.env.step(first_selected)

            while not done:
                selected, prob = self.model(state,greedy_action_selection)
                #(prob.size())
                #(prob_list.size())
                # shape: (batch, rollout)
                selected_clone = selected.clone()
                selected_clone[self.env.finished] = 0
                state, reward, done = self.env.step(selected_clone)

                prob_clone = prob.clone()
                prob_clone[self.env.finished] = 1
                prob_list = torch.cat((prob_list, prob_clone[:, :, None]), dim=2)
        #print(reward)
        if not self.tester_params['greedy_select_block']:
            #print("reward size",reward.size())
            #print(reward)
            reward = reward.reshape(batch_size * aug_factor, self.k, self.block_sample)
            #print("reward size", reward.size())
            #print(reward)
            reward_block = reward.mean(dim=2)
            #print("reward block",reward_block)
            reward_block_max,_ = reward.max(dim=2)
            reward_max,max_idx = reward_block.topk(1, dim=1)
            #print("reward max",reward_max)
            #print("max idx",max_idx)
            value,_ = reward_block_max.max(dim=1)
            #print("value",value)
        else:
            #最好
            #print(reward)
            _, max_idx = reward.topk(num_block, dim=1)
            reward_max, _ = reward.topk(1, dim=1)
            value = reward_max


            #最坏
            # a = -reward
            # _, max_idx = a.topk(1, dim=1)
            # value, _ = reward.topk(1, dim=1)

        #用多个块来抽样
        z_vectors_tensor = self.sample_vectors(max_idx, self.X, self.labels, num_vector_oneB)
        # print("z_size",z_vectors_tensor.size())
        #只用一个块来抽样
        # z_vectors_tensor = torch.zeros((max_idx.size(0), z_sample_size, 16))
        # for i, cluster_idx in enumerate(max_idx):
        #
        #     cluster_points = self.X[self.labels == cluster_idx.item()]
        #     if self.tester_params['sample_select']:
        #         #随机采样
        #         z_size = torch.from_numpy(cluster_points).size(dim=0)
        #         z_idx = torch.multinomial(
        #             (torch.ones(1, z_size) / z_size),
        #             z_sample_size, replacement=z_sample_size > z_size).cpu()
        #
        #         z_vectors_tensor[i] = torch.from_numpy(cluster_points[z_idx])
        #     else:
        #         #计算这些点到聚类中心的距离
        #         distances = np.linalg.norm(cluster_points - self.centroids[cluster_idx], axis=1)
        #         # # 找到距离最小的点的索引
        #         tensor_distances = torch.from_numpy(-distances)
        #         closest_point,closest_idx = (tensor_distances).topk(z_sample_size)
        #         # 添加这个点到closest_vectors列表中
        #         z_vectors_tensor[i] = torch.from_numpy(cluster_points[closest_idx])


        with torch.no_grad():
            self.env.rollout_size = self.rollout
            self.env.get_dataset_problem_OURS(self.data_file, batch_size, self.begin_idx, self.env_params['capacity'],
                                              aug_factor)
            self.begin_idx += batch_size
            reset_state, _, _ = self.env.reset()

            #z = torch.from_numpy(z_vectors)
            z =z_vectors_tensor
            # z = z.unsqueeze(0).repeat(aug_factor,1, 1, 1).reshape(batch_size * aug_factor, starting_points,z_sample_size, z_dim)
            # z = z.to(device)
            z = z.reshape(batch_size * aug_factor, rollout_size, z_dim)
            #print("finnal z ",z.size())
            self.model.pre_forward(reset_state, z)

            prob_list = torch.zeros(size=(self.env.batch_size, self.env.rollout_size, 0))

            # shape: (batch, rollout, 0~problem)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()

            # @#
            # prob = torch.ones(size=(batch_size, self.env.Bit_size))
            first_selected = torch.zeros(size=(batch_size * aug_factor, self.env.rollout_size))
            state, reward, done = self.env.step(first_selected)
            while not done:
                selected, prob = self.model(state, self.tester_params["greedy_action_selection_2"])
                # (prob.size())
                # (prob_list.size())
                # shape: (batch, rollout)
                selected_clone = selected.clone()
                selected_clone[self.env.finished] = 0
                state, reward, done = self.env.step(selected_clone)

                prob_clone = prob.clone()
                prob_clone[self.env.finished] = 1
                prob_list = torch.cat((prob_list, prob_clone[:, :, None]), dim=2)
        aug_reward = reward.reshape(aug_factor, batch_size, rollout_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        #print(max_pomo_reward)
        #本行为额外添加行
        no_aug_reward = -max_pomo_reward.mean(dim=0)
        # shape: (augmentation, batch)
        #no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        max_aug_pomo_reward = -max_aug_pomo_reward
        # shape: (batch,)
        #aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
        vector_no_aug_score = -max_pomo_reward[0, :].float().mean()
        vector_aug_score = -max_aug_pomo_reward.float().mean()
        # print(vector_aug_score)

        aug_greedy_reward = value.reshape(aug_factor, batch_size)
        #print(aug_greedy_reward)
        no_aug_greedy_reward = -aug_greedy_reward.mean(dim=0).reshape(-1)
        #no_aug_greedy_score = -aug_greedy_reward[0, :].float().mean()
        max_aug_greedy_reward, _ = aug_greedy_reward.max(dim=0)
        max_aug_greedy_reward = -max_aug_greedy_reward.reshape(-1)
        # aug_greedy_score = -max_aug_greedy_reward.float().mean()
        # print("aug greedy score",aug_greedy_score)
        condition_no = no_aug_reward < no_aug_greedy_reward  # 这会生成一个与 a 和 b 形状相同的布尔张量
        # print(condition_no)
        condition_aug = max_aug_pomo_reward < max_aug_greedy_reward
        # print(condition_aug)
        # print(max_aug_pomo_reward)
        # print(max_aug_greedy_reward)
        min_no_aug_reward = torch.where(condition_no, no_aug_reward, no_aug_greedy_reward)  # 根据条件选择 a 或 b 的元素
        min_aug_reward = torch.where(condition_aug, max_aug_pomo_reward, max_aug_greedy_reward)
        no_aug_score = min_no_aug_reward.mean()
        aug_score = min_aug_reward.float().mean()

        # print(aug_score)
        return no_aug_score.item(), aug_score.item()
        #return vector_no_aug_score.item(), vector_aug_score.item()

    # def bisecting_kmeans(X, k, max_iter=300, random_state=None):
    #     # 初始化所有点为一个簇
    #     clusters = [X]
    #
    #     # 递归划分簇直到达到所需的簇数量
    #     while len(clusters) < k:
    #         # 找到最大的簇进行划分
    #         max_cluster_idx = np.argmax([len(c) for c in clusters])
    #         max_cluster = clusters[max_cluster_idx]
    #
    #         # 使用KMeans算法划分最大的簇
    #         kmeans = KMeans(n_clusters=2, max_iter=max_iter, random_state=random_state)
    #         kmeans.fit(max_cluster)
    #
    #         # 获取划分后的簇中心和簇标签
    #         centers = kmeans.cluster_centers_
    #         labels = kmeans.labels_
    #
    #         # 根据标签将簇划分为两个子簇
    #         clusters[max_cluster_idx] = [max_cluster[labels == 0], max_cluster[labels == 1]]
    #
    #         # 展开列表以包含新的子簇
    #         clusters = [subcluster for cluster in clusters for subcluster in
    #                     (cluster if not isinstance(cluster, list) else cluster)]
    #
    #         # 提取最终的簇中心和标签
    #     final_centers = [np.mean(cluster, axis=0) for cluster in clusters]
    #     final_labels = []
    #     for i, cluster in enumerate(clusters):
    #         final_labels.extend([i] * len(cluster))
    #     final_labels = np.array(final_labels)
    #
    #     # 将原始数据点的索引映射到最终的簇标签
    #     label_mapping = {point_id: label for label, cluster in enumerate(clusters) for point_id in range(len(X)) if
    #                      X[point_id] in cluster}
    #     final_labels = np.array([label_mapping[i] for i in range(len(X))])
    #
    #     return final_centers, final_labels

    def bisecting_kmeans(X, K,random_state=None):
        # 初始化聚类中心为随机样本点
        if random_state is not None:
            np.random.seed(random_state)
        centroids = [X[np.random.choice(len(X))]]

        # 递归进行二分
        def bisect(cluster_data, depth=0):
            if depth == K - 1:  # 达到目标簇数量，停止递归
                return [cluster_data]

                # 使用K-Means算法将当前簇分为两个子簇
            # 这里为了简化，我们直接使用sklearn的KMeans，但也可以自己实现K-Means算法
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(cluster_data)
            centroids.extend(kmeans.cluster_centers_)

            # 返回两个子簇的数据
            return [cluster_data[kmeans.labels_ == i] for i in range(2)]

            # 初始聚类

        clusters = [X]

        # 递归二分每个簇
        while len(clusters) < K:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:  # 避免对单个样本点进行划分
                    new_clusters.extend(bisect(cluster, depth=len(clusters)))
                else:
                    new_clusters.append(cluster)
            clusters = new_clusters

            # 返回最终的簇和聚类中心
        return clusters, np.array(centroids)

    def sample_vectors(self,a, X, labels, b):
        # 初始化结果张量，大小为 (batch_size, k*b, vector_dim)
        z = torch.zeros(a.size(0), a.size(1) * b, 16)

        # 使用循环遍历a中的每个实例和每个池索引
        for i in range(a.size(0)):
            # 处理每一个实例
            # z_vectors_block = torch.zeros((self.k, self.block_sample, 16))
            for j in range(a.size(1)):
                pool_index = a[i, j].item()  # 当前需要抽取的池索引

                # 从对应标签的向量中选取b个随机向量
                cluster_points = X[labels == pool_index]
                # 随机采样
                z_size = torch.from_numpy(cluster_points).size(dim=0)
                z_idx = torch.multinomial((torch.ones(1, z_size) / z_size), b, replacement=b > z_size).cpu()
                sampled_vectors = torch.from_numpy(cluster_points[z_idx])
                # print(sampled_vectors)

                # 将抽取的向量填充到z中
                start_idx = j * b
                end_idx = (j + 1) * b
                z[i, start_idx:end_idx] = sampled_vectors
                # print("z ",z)
        return z