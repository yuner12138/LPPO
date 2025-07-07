
"""
The MIT License

Copyright (c) 2020 POMO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass

import pandas as pd
import torch
import pickle

from PDTSProblemDef import get_random_problems, augment_xy_data_by_8_fold,augment_xy_data_by_8_fold_N2S,get_dataset_problem,aug


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    ROLLOUT_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class PDTSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.group_state = None
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.rollout_size = None

        self.at_the_depot = None

        self.FLAG__use_saved_problems = False
        self.saved_problems = None
        self.saved_index = 0

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.ROLLOUT_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, rollout)
        self.selected_node_list = None
        # shape: (batch, rollout, 0~problem)


    def use_saved_problems(self, filename):
        self.FLAG__use_saved_problems = True
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        self.saved_problems = torch.tensor(data)
        # print("save",self.saved_problems.size())
        self.saved_index = 0
    def use_pkl_saved_problems_OURs(self, filename, num_problems, index_begin=0):
        print("use me")
        self.FLAG__use_saved_problems = True

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        partial_data = list(data[i] for i in range(index_begin, index_begin+num_problems))
        arrays_101x2 = []
        for pair in partial_data:
            # 第一个子元素，长度为2的列表
            # first_part = pair[0]
            # print("first_part",first_part)
            # # 第二个子元素，长度为100的列表，每个元素是长度为2的元组
            # second_part = pair[1]
            # print("second_part",second_part)
            # # 将第一个子元素和第二个子元素合并成一个长度为101的列表
            # combined_part = [first_part] + [tuple_item for tuple_item in second_part]
            # 将合并后的列表转换为形状为(101, 2)的NumPy数组
            array_101x2 = np.array(pair).reshape(-1, 2)
            arrays_101x2.append(array_101x2)
        data_final = np.stack(arrays_101x2)
        tensor = torch.from_numpy(data_final)
        tensor = tensor.to(device='cuda')
        tensor = tensor.float()
        self.saved_problems = tensor
        #self.saved_index = 0
        # print("save", self.saved_problems.size())
    def use_pkl_saved_problems(self, filename, num_problems, index_begin=0):
        self.FLAG__use_saved_problems = True

        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        partial_data = list(data[i] for i in range(index_begin, index_begin+num_problems))
        arrays_101x2 = []
        for pair in partial_data:
            # 第一个子元素，长度为2的列表
            first_part = pair[0]
            # print("first_part",first_part)
            # 第二个子元素，长度为100的列表，每个元素是长度为2的元组
            second_part = pair[1]
            # print("second_part",second_part)
            # 将第一个子元素和第二个子元素合并成一个长度为101的列表
            combined_part = [first_part] + [tuple_item for tuple_item in second_part]
            # 将合并后的列表转换为形状为(101, 2)的NumPy数组
            array_101x2 = np.array(combined_part).reshape(-1, 2)
            arrays_101x2.append(array_101x2)
        data_final = np.stack(arrays_101x2)
        tensor = torch.from_numpy(data_final)
        tensor = tensor.to(device='cuda')
        tensor = tensor.float()
        self.saved_problems = tensor
        #self.saved_index = 0
        # print("save", self.saved_problems.size())

    def use_random_problems(self):
        self.FLAG__use_saved_problems = False

        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None

    def load_problems(self, batch_size, rollout_size, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size

        if not self.FLAG__use_saved_problems:
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            self.problems = self.saved_problems[self.saved_index:self.saved_index+batch_size]
            # print("probem",self.problems.size(0))
            self.saved_index += batch_size

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor

            self.problems = augment_xy_data_by_8_fold_N2S(self.problems,aug_factor)
            # shape: (8*batch, problem, 2)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)

    def load_problems_repeat(self, batch_size, problems,rollout_size, aug_factor=1):
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        self.problems = problems
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8

                self.problems = augment_xy_data_by_8_fold_N2S(self.problems,aug_factor)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)
    def load_dataset_problem(self, batch_size, sample, aug_type='8'):
        if sample:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        depot_x_y, pick_x_y, delivery_x_y, pick_demand, delivery_demand, customer_size, capacity, data, aug_number = self.get_dataset_problem(
            load_path=self.load_path,
            batch_size=self.batch_size,
            aug_type=aug_type
        )
        depot_x_y = depot_x_y.to('cuda:0')
        pick_x_y = pick_x_y.to('cuda:0')
        delivery_x_y = delivery_x_y.to('cuda:0')
        pick_demand = pick_demand.to('cuda:0')
        delivery_demand = delivery_demand.to('cuda:0')
        self.customer_size = customer_size
        self.rollout_size = customer_size
        self.batch_size = self.batch_size * aug_number

        self.problems = torch.cat((depot_x_y, pick_x_y, delivery_x_y), dim=1)
        # shape: (batch, node, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        self.all_node_demand = torch.cat((depot_demand, pick_demand, delivery_demand), dim=1)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)

        self.reset_state.depot_x_y = depot_x_y
        self.reset_state.pick_x_y = pick_x_y
        self.reset_state.delivery_x_y = delivery_x_y
        self.reset_state.pick_demand = pick_demand
        self.reset_state.delivery_demand = delivery_demand
        self.reset_state.customer_size = pick_x_y.size(1)
        self.reset_state.capacity = capacity

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX
        return data
    def get_dataset_problem(self,load_path, batch_size,rollout_size, aug_type):

        self.rollout_size = rollout_size
        filename = load_path
        data = pd.read_csv(filename, sep=',', header=None)
        data = data.to_numpy()
        depot_size = 1
        customer_size = int(data[0][0])
        scale = int(data[0][1])
        depot_xy = data[1:depot_size + 1] / scale
        pick_xy = data[depot_size + 1:depot_size + customer_size + 1] / scale
        delivery_xy = data[depot_size + customer_size + 1:depot_size + 2 * customer_size + 1] / scale
        full_node = data[1:depot_size + 2 * customer_size + 1] / scale

        depot_x_y = torch.FloatTensor(depot_xy[0][0:2]).unsqueeze(0)
        for i in range(len(2 * pick_xy)):
            pick_x_y = torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
                [pick_x_y, torch.FloatTensor(pick_xy[i][0:2]).unsqueeze(0)], dim=0)
            delivery_x_y = torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0) if i == 0 else torch.cat(
                [delivery_x_y, torch.FloatTensor(delivery_xy[i][0:2]).unsqueeze(0)], dim=0)
        depot_x_y = depot_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
        pick_x_y = pick_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
        delivery_x_y = delivery_x_y.unsqueeze(0).repeat(batch_size, 1, 1)
        # print("aug_num",aug_type)
        # print(data)
        depot_x_y, pick_x_y, delivery_x_y, aug_number = self.aug(
            aug_type=aug_type, depot_x_y=depot_x_y, pick_x_y=pick_x_y, delivery_x_y=delivery_x_y)
        # print("aug_num", aug_number)
        data = {'depot_x_y': depot_x_y.numpy().tolist(), 'pick_x_y': pick_x_y.numpy().tolist(),
                'delivery_x_y': delivery_x_y.numpy().tolist(),
                'full_node': full_node, 'scale': scale, 'aug_number': aug_number}
        # print(data['full_node'])
        # print("depot",depot_x_y.size())
        data = data
        self.batch_size = batch_size*aug_number
        self.problems = torch.cat((depot_x_y,pick_x_y,delivery_x_y),dim=1).to(device = 'cuda')
        # print("problem",self.problems)
        return depot_x_y, pick_x_y, delivery_x_y, customer_size, data, aug_number

    def aug(self,aug_type, depot_x_y, pick_x_y, delivery_x_y):
        aug_number = 1
        if aug_type == 8:
            aug_number = 8
            depot_x_y = self.augment_x_y_by_8(depot_x_y)
            pick_x_y = self.augment_x_y_by_8(pick_x_y)
            delivery_x_y = self.augment_x_y_by_8(delivery_x_y)
        elif aug_type == 9:
            aug_number = 9
            pick_x_y_2 = delivery_x_y.clone()
            delivery_x_y_2 = pick_x_y.clone()
            depot_x_y_1 = self.augment_x_y_by_8(depot_x_y)
            depot_x_y_2 = depot_x_y.clone()
            pick_x_y_1 = self.augment_x_y_by_8(pick_x_y)
            delivery_x_y_1 = self.augment_x_y_by_8(delivery_x_y)

            depot_x_y = torch.cat((depot_x_y_1, depot_x_y_2), dim=0)
            pick_x_y = torch.cat((pick_x_y_1, pick_x_y_2), dim=0)
            delivery_x_y = torch.cat((delivery_x_y_1, delivery_x_y_2), dim=0)
        return depot_x_y, pick_x_y, delivery_x_y, aug_number

    def augment_x_y_by_8(self,x_y):
        # shape: (batch, N, 2)
        x = x_y[:, :, [0]]
        y = x_y[:, :, [1]]
        # shape: (batch, N, 1)

        data1 = torch.cat((x, y), dim=2)
        data2 = torch.cat((1 - x, y), dim=2)
        data3 = torch.cat((x, 1 - y), dim=2)
        data4 = torch.cat((1 - x, 1 - y), dim=2)
        data5 = torch.cat((y, x), dim=2)
        data6 = torch.cat((1 - y, x), dim=2)
        data7 = torch.cat((y, 1 - x), dim=2)
        data8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_x_y = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
        # shape: (8 * batch_size, N, 2)
        return aug_x_y
    def reset(self):
        # print("problem",self.problems.size())
        distance = compute_euclidean_distances(self.problems)
        self.group_state = GROUP_STATE(group_size=self.rollout_size, data=self.problems, distance = distance)
        self.group_state.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.rollout_size)
        self.group_state.ROLLOUT_IDX = torch.arange(self.rollout_size)[None, :].expand(self.batch_size, self.rollout_size)

        self.selected_count = 0
        self.current_node = None
        # shape: (batch, rollout)
        self.selected_node_list = torch.zeros((self.batch_size, self.rollout_size, 0), dtype=torch.long)
        # shape: (batch, rollout, 0~problem)

        # CREATE STEP STATE
        # self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, ROLLOUT_IDX=self.ROLLOUT_IDX)
        # self.step_state.ninf_mask = torch.zeros((self.batch_size, self.rollout_size, self.problem_size))
        # shape: (batch, rollout, problem)

        self.group_state.ninf_mask = torch.zeros((self.batch_size, self.rollout_size, self.problem_size))

        reward = None
        done = False
        return Reset_State(self.problems),self.group_state, reward, done

    def reset_state(self):  # 返回生成的问题，重置奖励为None，done为False，并将state置初始状态
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.rollout_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
       # self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, Bit_IDX=self.Bit_IDX)
        self.group_state.BATCH_IDX = self.BATCH_IDX
        self.group_state.ROLLOUT_IDX = self.ROLLOUT_IDX
        # 设置当前步骤的状态
        # BATCH第一行为0，第二行为1，最后一行值为batch-1.每行代表一个数据组的POMO
        # POMO每行为0->POMO-1，一共batch行值均如此
        # Step_State为一个状态class，包括BATCH——idx，POMO_IDX,当前节点和mask
        #self.step_state.ninf_mask = torch.zeros((self.batch_size, self.Bit_size, self.problem_size))
        self.group_state.ninf_mask = torch.zeros((self.batch_size, self.rollout_size, self.problem_size))

    def pre_step(self):
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, rollout)
        self.group_state.move_to(selected)

        # self.selected_count += 1
        # self.current_node = selected
        # # shape: (batch, rollout)
        # self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # # shape: (batch, rollout, 0~problem)
        #
        # # UPDATE STEP STATE
        # self.step_state.current_node = self.current_node
        # # shape: (batch, rollout)
        # self.step_state.ninf_mask[self.BATCH_IDX, self.ROLLOUT_IDX, self.current_node] = float('-inf')
        # shape: (batch, rollout, node)

        # returning values
        done = self.group_state.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.group_state, reward, done

    def _get_travel_distance1(self):

        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(self.batch_size, -1,
                                                                                  self.problem_size, 2)
        # shape: (batch, Bit, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.rollout_size, self.problem_size, 2)
        # shape: (batch, Bit, problem, 2)
        gathering_index = gathering_index.long()
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, Bit, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, Bit, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, Bit)
        return travel_distances

    def _get_travel_distance(self):
        all_node_xy = self.problems[:, None, :, 0:2].expand(self.batch_size, self.rollout_size, -1, 2)
        # print("all_node",all_node_xy)
        # print(all_node_xy.size())
        # shape = (batch, group, problem+1, 2)
        gathering_index = self.group_state.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape = (batch, group, selected_count, 2)
        gathering_index = gathering_index.long()
        ordered_seq = all_node_xy.gather(dim=2, index=gathering_index)
        # shape = (batch, group, selected_count, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # size = (batch, group, selected_count)

        travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return travel_distances
class GROUP_STATE:

    def __init__(self, group_size, data,distance):
        # datasets.shape = (batch, problem+1, 3)
        self.distance = distance
        self.problem = data.size(1)
        self.BATCH_IDX: torch.Tensor
        self.ROLLOUT_IDX: torch.Tensor
        # shape: (batch, Bit)
        # shape: (batch, Bit)
        # shape: (batch, Bit, node)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        device = torch.device("cuda")
        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        ByteTensor = torch.cuda.ByteTensor
        BoolTensor = torch.cuda.BoolTensor
        Tensor = FloatTensor
        # print("batch",self.batch_s)
        # print("group",self.group_s)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, self.group_s, 0)))
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.at_the_depot = None
        # shape = (batch, group)
        self.to_delivery = torch.cat([torch.ones(self.batch_s, self.group_s, (self.problem-1) // 2 + 1, dtype=torch.uint8, device=device),
            torch.zeros(self.batch_s, self.group_s, (self.problem-1) // 2, dtype=torch.uint8, device=device)], dim=-1)
        # [batch_size, group_s, graph_size+1], [1,1...1, 0...0]
        self.visited_ninf_flag = Tensor(np.zeros((self.batch_s, self.group_s, (self.problem-1)+1)))
        # shape = (batch, group, problem+1)
        self.ninf_mask = Tensor(np.zeros((self.batch_s, self.group_s, (self.problem-1)+1)))
        # shape = (batch, group, problem+1)
        self.finished = BoolTensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)

    def move_to(self, selected):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, Bit)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # print("selected_node_list",self.selected_node_list)
        # print(self.selected_node_list.size())
        # shape: (batch, Bit, 0~problem)
        # selected_node_list,是每次选出来的点的标号的列表，每行即为一个实例的一个解轨迹

        # UPDATE STEP STATE

        # shape: (batch, Bit)
        self.ROLLOUT_IDX = self.ROLLOUT_IDX.long()
        self.BATCH_IDX = self.BATCH_IDX.long()
        self.current_node = self.current_node.long()
        selected = selected.to(torch.long)

        self.at_the_depot = (selected == 0)
        new_to_delivery = (selected + (self.problem - 1) // 2) % (self.problem)

        self.visited_ninf_flag[self.BATCH_IDX, self.ROLLOUT_IDX, selected] = -np.inf

        self.to_delivery = self.to_delivery.scatter(-1, new_to_delivery[:, :, None], 1)
        self.finished = self.finished + (self.visited_ninf_flag == -np.inf).all(dim=2)

        self.ninf_mask = (
                    (self.visited_ninf_flag.clone() == -np.inf).type(torch.uint8) | (1 - self.to_delivery)).float()

        self.ninf_mask[self.ninf_mask == 1] = -np.inf

        self.ninf_mask[self.finished[:, :, None].expand(self.batch_s, self.group_s, self.problem)] = 0

def compute_euclidean_distances(a):
    """
    计算张量a中每个实例内部点之间的欧式距离
    :param a: 输入张量，形状为 (batch, problem_size, 2)
    :return: 距离张量，形状为 (batch, problem_size, problem_size)
    """
    # 计算每个点的二维坐标的平方
    # print("a",a)
    a_squared = a ** 2

    # 增加维度以准备广播
    # a_expanded 的形状为 (batch, problem_size, problem_size, 2)
    a_expanded = a[:, :, None, :].expand(a.size(0), a.size(1), a.size(1), a.size(2))

    # b_expanded 也需要相同的形状
    b_expanded = a[:, None, :, :].expand(a.size(0), a.size(1), a.size(1), a.size(2))

    # 计算两点之间坐标差的平方
    squared_diff = (a_expanded - b_expanded) ** 2

    # 对坐标差的平方求和（沿最后一个维度，即坐标维度）
    distances_squared = squared_diff.sum(-1)

    # 注意：这里得到的距离矩阵是对称的，且对角线上的元素为0（点到自身的距离）
    # 如果需要，可以在这一步添加任何必要的后处理，比如将对角线设置为一个很小的数以避免除以零

    # 开方得到欧式距离
    distances = distances_squared.sqrt()

    # 由于距离矩阵是对称的，你可以选择只保留上三角或下三角部分，但这取决于你的需求
    # 这里我们直接返回整个矩阵

    return distances
