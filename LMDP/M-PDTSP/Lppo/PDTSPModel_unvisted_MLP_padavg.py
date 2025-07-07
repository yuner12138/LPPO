import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class PDTSPModel_unvisted_MLP_padavg(nn.Module):

    def __init__(self, model_params,
                 trainer_params):
        super().__init__()
        self.model_params = model_params
        self.force_first_move = model_params['force_first_move']

        self.encoder = PDTSP_Encoder(**model_params)
        self.decoder = PDTSP_Decoder(**model_params)
        # self.collect = PDTSP_Collect(**model_params)
        self.encoded_nodes = None
        self.encoded_graph = None
        # shape: (batch, problem, EMBEDDING_DIM)
        self.k_near = model_params['max_unvisited']
        self.start_q1 = nn.Parameter(torch.zeros(model_params['embedding_dim']), requires_grad=True)
        self.start_last_node = nn.Parameter(torch.zeros(model_params['embedding_dim']), requires_grad=True)

    def pre_forward(self, reset_state, z):
        # print("in_encoder")
        self.encoded_nodes = self.encoder(reset_state)
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes, z)


    def get_expand_prob(self, state):

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        # shape: (batch, beam_width, embedding)
        probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
        # shape: (batch, beam_width, problem)

        return probs

    # 强制从不同起点开始
    def forward(self, state, greedy_construction=False, EAS_incumbent_action=None):
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)
        problem_size = self.encoded_nodes.size(1)
        # print("problem_size",problem_size)

        if self.force_first_move and state.current_node is None:
            selected = torch.arange(problem_size).repeat(rollout_size // problem_size)[None, :].expand(batch_size,
                                                                                                       rollout_size)
            prob = torch.ones(size=(batch_size, rollout_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:  # 全0开始，随机一个占位符去选择第一个点
            if state.current_node is None:
                # print("第一个点")
                self.decoder.set_q1(self.start_q1[None, None].expand(batch_size, rollout_size, -1))
                encoded_last_node = self.start_last_node[None, None].expand(batch_size, rollout_size, -1)
                depots = torch.zeros(batch_size, rollout_size)
                # 可能有损随机性
                clust_labels = _get_encoding(self.decoder.cluster_labels, depots)
            else:
                # print("非第一个点")
                # print(state.current_node)
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                # print("labels",self.decoder.cluster_labels.size())
                # clust_labels = _get_encoding(self.decoder.cluster_labels, state.current_node)

                # print("labels", clust_labels.size())
            # shape: (batch, rollout, embedding)
            # print("in_decoder")
            probs = self.decoder(self.encoded_graph, self.encoded_nodes, encoded_last_node, state, self.k_near,
                                 ninf_mask=state.ninf_mask)
            # shape: (batch, rollout, problem+1)

            if not greedy_construction:
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        # print("probs",probs.size())
                        selected = probs.reshape(batch_size * rollout_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, rollout_size)
                    # shape: (batch, rollout)

                    if EAS_incumbent_action is not None:
                        selected[:, -1] = EAS_incumbent_action

                    prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
                    # shape: (batch, rollout)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, rollout)
                prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
            if state.current_node is None:
                encoded_first_node = _get_encoding(self.encoded_nodes, selected)
                # shape: (batch, pomo, embedding)
                self.decoder.set_q1(encoded_first_node)
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, rollout)

    batch_size = node_index_to_pick.size(0)
    rollout_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, rollout_size, embedding_dim)
    # shape: (batch, rollout, embedding)
    gathering_index = gathering_index.long()
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, rollout, embedding)

    return picked_nodes


def _get_encoding_pad0(encoded_nodes, node_index_to_pick, k):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, rollout)

    batch_size = node_index_to_pick.size(0)
    rollout_size = int(node_index_to_pick.size(1) / k)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, rollout_size * k, embedding_dim)
    # shape: (batch, rollout, embedding)
    gathering_index = gathering_index.long()
    zeros_tensor = torch.zeros(batch_size, 1, embedding_dim)
    # 沿着第二个维度（problem维度）连接a和zeros_tensor
    b = torch.cat((encoded_nodes, zeros_tensor), dim=1)
    # print("b",b)
    unvist_node_emb = b.gather(dim=1, index=gathering_index)
    # shape: (batch, rollout, embedding)

    return unvist_node_emb


########################################
# ENCODER
########################################


class PDTSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.init_embed_pick_demand = nn.Linear(6, embedding_dim)
        self.init_embed_delivery_demand = nn.Linear(3, embedding_dim)

        # self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, reset_state):
        # datasets.shape: (batch, problem, 2)
        problem = self.model_params['problem_size']
        # datasets.shape: (batch, problem, 2)
        depot_xy = reset_state.depot_x_y  # 取出仓库节点位置
        # depot_xy = depot_xy.long()
        # shape = (batch, 1, 2)
        depot_xy = depot_xy.to(self.embedding_depot.weight.dtype)
        pick_x_y = reset_state.pick_x_y
        delivery_x_y = reset_state.delivery_x_y
        pick_demand = reset_state.pick_demand
        delivery_demand = reset_state.delivery_demand
        node_xy = torch.cat(( pick_x_y, delivery_x_y), dim=1)  # 后续节点位置
        # shape = (batch, problem, 2)

        pick_x_y_demand = torch.cat((pick_x_y, pick_demand[:, :, None]), dim=2)
        # print("pick",pick_x_y_demand.size())
        delivery_x_y_demand = torch.cat((delivery_x_y, delivery_demand[:, :, None]), dim=2)
        node_xy_demand = torch.cat(( pick_x_y_demand, delivery_x_y_demand), dim=1)
        
        embed_depot = self.embedding_depot(depot_xy)  # 得到仓库的嵌入

        # [batch_size, graph_size//2, 4] loc
        # print("node_xy_demand",node_xy_demand.size())
        # print(node_xy_demand[:, :(problem - 1) // 2, :].size())
        feature_pick = torch.cat([node_xy_demand[:, :(problem - 1) // 2, :], node_xy_demand[:, (problem - 1) // 2:, :]], -1)
        # 将node_xy沿第二个维度上分为两个部分，再在第三个维度上将这两个部分拼接
        # [batch_size, graph_size//2, 2] loc
        feature_delivery = node_xy_demand[:, (problem - 1) // 2:, :]  # [batch_size, graph_size//2, 2]
        feature_pick = feature_pick.to(self.init_embed_pick_demand.weight.dtype)
        embed_pick = self.init_embed_pick_demand(feature_pick)
        feature_delivery = feature_delivery.to(self.init_embed_delivery_demand.weight.dtype)
        embed_delivery = self.init_embed_delivery_demand(feature_delivery)

        out = torch.cat([embed_depot, embed_pick, embed_delivery], 1)
        # embedded_input = self.embedding(datasets)
        # # shape: (batch, problem, embedding)
        #
        # out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.multi_head_attn = MultiHeadAttention(**model_params)
        # self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = self.multi_head_attn(input1)  # 多头注意力机制编码过程，通过输入进行一次h1->h2的过程
        # qkv shape: (batch, head_num, problem, qkv_dim)

        # out_concat = self.attention_fn(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class PDTSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding_dim = embedding_dim
        lppo_embedding_dim = self.model_params['lppo_embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        z_dim = model_params['z_dim']
        self.problem_size = self.model_params['problem_size']
        # self.rollouts_size = trainer_params['train_z_sample_size']

        self.use_EAS_layers = False

        # self.num_clusters = self.model_params['em_clusters']
        self.max_unvisited = self.model_params['max_unvisited']
        self.em_iter = self.model_params['n_iter']

        self.Wq = nn.Linear(2 * embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq_clu = nn.Linear(self.embedding_dim * 2, head_num * qkv_dim, bias=False)

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.hyper_Wq_last = DiagonalMLP(self.embedding_dim + self.embedding_dim,
                                         self.embedding_dim, self.embedding_dim)
        self.cluster_embed = EM_LayerCont(self.embedding_dim, self.embedding_dim, num_clusters=self.max_unvisited,
                                          num_iter=self.em_iter)
        k_near = self.model_params['max_unvisited']
        self.unvisited_MLP = UnvisitedMLP(self.embedding_dim * k_near, self.embedding_dim * 5, self.embedding_dim)
        self.cluster_combine = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        self.z = None  # saved z vector for decoding
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

        self.lppo_layer_1 = nn.Linear(embedding_dim + z_dim, lppo_embedding_dim)
        self.lppo_layer_2 = nn.Linear(lppo_embedding_dim, embedding_dim)
        # 初始化权重为0
        # init.zeros_(self.lppo_layer_1.weight)
        # init.zeros_(self.lppo_layer_2.weight)

        # 注意：如果你也想初始化偏置为0，可以这样做
        # init.zeros_(self.lppo_layer_1.bias)
        # init.zeros_(self.lppo_layer_2.bias)
        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention
        self.count = 0

    def set_kv_C(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

        # shape: (batch, rollout, z_dim)

    # POMO
    # def set_kv(self, encoded_nodes, z):
    #     # encoded_nodes.shape: (batch, problem+1, embedding)
    #     head_num = self.model_params['head_num']
    #
    #     self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
    #     self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
    #     # shape: (batch, head_num, problem+1, qkv_dim)
    #     self.single_head_key = encoded_nodes.transpose(1, 2)
    #     # shape: (batch, embedding, problem+1)
    #     self.z = z
    #     # shape: (batch, rollout, z_dim)

    # cluster
    def set_kv(self, encoded_nodes, z):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # create the cluster embeddings
        # exclude the depot from being a part of this clustering - we should optimize for the non-depot nodes

        # _, cluster_reps, scores = self.cluster_embed(encoded_nodes)
        # （num_cluster,batch,2*batch）
        # calculate the loss and keep aside
        # |cluster_reps| = (batch, pomo, num_cluster, embedding)

        # self.cluster_reps = cluster_reps.unsqueeze(1).repeat(1, z.size(1), 1, 1)
        # |self.cluster_scores| = (batch, n_nodes, cluster_number)

        # self.cluster_labels = scores.permute(0, 2, 1)

        # change the key and value stores to be the gnn representations instead - each rep is now contextualized with
        # its neighborhood
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.permute(0, 2, 1)
        # shape: (batch, embedding, problem)

        # average embedding of all unvisited cities, this will remove the current embedding at each step as the decoder moves
        # shape: (batch, pomo, embedding)
        # all starting points have this
        self.num_unvisited = encoded_nodes.size(1)
        self.unvisited_cities = encoded_nodes.mean(dim=1)[:, None, :].repeat(1, self.num_unvisited, 1)
        self.z = z

    def set_z(self, z):
        self.z = z

    def _update_unvisited(self, embedding, cluster_weights):
        # update the cluster centers
        # embedding = (batch, pomo, embedding)
        # cluster_weights = (batch, pomo, num_cluster)
        # cluster_reps = (batch, pomo, num_cluster, embedding)
        cluster_weights = cluster_weights.unsqueeze(-1).repeat(1, 1, 1, self.embedding_dim)

        embedding = embedding.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        self.cluster_reps = self.cluster_reps - torch.multiply(cluster_weights, embedding)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo

        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def reset_EAS_layers(self, batch_size):
        self.EAS_W1 = torch.nn.Parameter(self.lppo_layer_1.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b1 = torch.nn.Parameter(self.lppo_layer_1.bias.repeat(batch_size, 1))
        self.EAS_W2 = torch.nn.Parameter(self.lppo_layer_2.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b2 = torch.nn.Parameter(self.lppo_layer_2.bias.repeat(batch_size, 1))
        self.use_EAS_layers = True

    def get_EAS_parameters(self):
        return [self.EAS_W1, self.EAS_b1, self.EAS_W2, self.EAS_b2]

    def forward(self, all_embedding, encode_emb, encoded_last_node, state, k, ninf_mask):
        # all embedding shape: (batch, 1, EMBEDDING_DIM)
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # print("ninf",ninf_mask)
        head_num = self.model_params['head_num']

        # input_cat = torch.cat((all_embedding.expand(-1, encoded_last_node.size(dim=1), -1), encoded_last_node), dim=2)
        # size(batch,rollout,2*embedding)
        # self._update_unvisited(encoded_last_node, node_cluster_embed)

        # use the GNN-focused representations instead
        b, p, emb = encoded_last_node.size()

        # input_cat = torch.cat((encoded_last_node, self.cluster_reps.view(b, p, self.num_clusters * emb)), dim=2)
        # print("input_cat",input_cat.size())
        # 此处为当前节点信息和聚类表示信息
        # size(batch,rollout,(num_clusters+1)*embedding)
        #  Multi-Head Attention
        #######################################################
        # q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        unvist_node_index = search_unvisited(state.current_node, state.distance, k, ninf_mask)
        # print("unvist_node_index",unvist_node_index)
        # print(unvist_node_index.size())
        unvist_node_index = unvist_node_index.reshape(b, -1)
        # print("in-get_encoding_pad0")

        start_time = time.time()
        unvist_node_emb = _get_encoding_pad0(encode_emb, unvist_node_index, k)
        # print("out-get_encoding_pad0")
        end_time = time.time()
        # print(f"Execution time: {end_time - start_time} seconds")

        unvist_node_emb = unvist_node_emb.reshape(b, p, k, emb)
        # print("in-gather_pad")

        start_time = time.time()
        unvist_node_emb = gather_PAD_AVG(unvist_node_emb)
        # print("out-gather_pad")
        end_time = time.time()
        # print(f"Execution time: {end_time - start_time} seconds")

        # print("ceshi",unvist_node_emb)
        unvist_node_emb = unvist_node_emb.reshape(b, p, -1)
        # print("unvist_node_emb",unvist_node_emb.size())
        unvist_node_emb_MLP = self.unvisited_MLP(unvist_node_emb)
        # print("unvist_node_emb", unvist_node_emb)
        # print(unvist_node_emb.size())
        # if state.current_node <= self.problem_size/2:
        #     deliver_node = state.current_node+50
        #     deliver_node_emb = _get_encoding(self.encoded_nodes, deliver_node)
        #     emphasize_cat = torch.cat((encoded_last_node, deliver_node_emb,unvist_node_emb,1), dim=2)
        # elif state.current_node > self.problem_size/2:
        #     emphasize_cat = torch.cat((encoded_last_node, unvist_node_emb,0), dim=2)
        # else:
        #     emphasize_cat = input_cat

        emphasize_cat = torch.cat((encoded_last_node, unvist_node_emb_MLP), dim=2)
        # print(emphasize_cat.size())
        # print("emphasize_cat",emphasize_cat)
        q = reshape_by_heads(self.Wq_clu(emphasize_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        # print("q",q)
        b, p, emb = encoded_last_node.size()
        # print("k",self.k)
        # print("v",self.v)
        out_concat = self.attention_fn(q, self.k, self.v, rank3_ninf_mask=ninf_mask)

        # shape: (batch, rollout, head_num*qkv_dim)
        # print("out_concat",out_concat)
        mh_atten_out = self.multi_head_combine(out_concat)
        # print("mh_atten_out", mh_atten_out)
        # shape: (batch, rollout, embedding)
        # if padding

        weights = self.hyper_Wq_last(emphasize_cat)
        # print("weights",weights)
        mh_atten_out = mh_atten_out * weights
        # print("mh_atten_out",mh_atten_out)
        # print("self.z",self.z)
        if not self.use_EAS_layers:
            # print("z",self.z.size())
            lppo_out = self.lppo_layer_1(torch.cat((mh_atten_out, self.z), dim=2))
            # print("lppo_out_1",lppo_out)
            # shape: ?
            lppo_out = F.relu(lppo_out)
            # print("lppo_out_relu", lppo_out)
            # shape: ?
            lppo_out = self.lppo_layer_2(lppo_out)
            # print("lppo_out_2", lppo_out)
            # shape: ?
        else:
            lppo_out = torch.matmul(torch.cat((mh_atten_out, self.z), dim=2), self.EAS_W1)
            lppo_out += self.EAS_b1[:, None]
            # shape: ?
            lppo_out = F.relu(lppo_out)
            # shape: ?
            lppo_out = torch.matmul(lppo_out, self.EAS_W2)
            # shape: ?
            lppo_out += self.EAS_b2[:, None]

        # print("Lppo",lppo_out)
        mh_atten_out += lppo_out
        # print("mh_atten_out",mh_atten_out)
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, rollout, problem)
        # print("score",score)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, rollout, problem)
        # print("score_scaled",score_scaled)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # print("score_clipped",score_clipped)
        score_masked = score_clipped + ninf_mask
        # print("score_masked",score_masked)
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, rollout, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################
class EM_LayerCont(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_clusters, num_iter):
        super().__init__()
        self.num_iter = num_iter
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # nn.embedding()按照对应的行号，选出对应的聚类的向量表示。
        self.clusters = nn.Embedding(num_clusters, hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.clust_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # self.in_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.in_attn = nn.MultiheadAttention(hidden_dim, num_heads=1)

        self.norm_layer = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)

    def forward(self, h):
        clust_reps = self.clusters(torch.arange(self.num_clusters))[None, :, :].repeat(h.size(0), 1, 1)
        h = h.permute(1, 0, 2)
        clust_reps = self.clust_proj(clust_reps)
        h = self.in_proj(h)

        in_emb = clust_reps
        in_emb = in_emb.permute(1, 0, 2)
        for i in range(self.num_iter):
            clust_reps, weights = self.in_attn(in_emb, h, h)
            # in_emb(batch,num_clusters,embedding)
            # print("h",h.size())
            # print("in_emb",in_emb.size())
            # print("clust_reps",clust_reps.size())
            # print("weights",weights.size())
            # a = weights @ (h.permute(1,0,2))
            # print("a ",a.size())
            # print(a==clust_reps.permute(1,0,2))
            # weights(batch,head_size,seq_len,seq_len)
            # h(batch,problem,embeddings)
            # clust_reps = weights @ h
            in_emb = in_emb + clust_reps
            # in_emb = self.norm_layer(in_emb.transpose(1, 2)).transpose(1, 2)
            in_emb = self.norm_layer(in_emb.transpose(1, 2)).transpose(1, 2)
        # |weights| = (batch, cluster_number, n_nodes)
        ##（num_cluster,batch,2*batch）
        return None, in_emb.permute(1, 0, 2), weights


class EM_LayerCont_old(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_clusters, num_iter):
        super().__init__()
        self.num_iter = num_iter
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # nn.embedding()按照对应的行号，选出对应的聚类的向量表示。
        self.clusters = nn.Embedding(num_clusters, hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.clust_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.in_attn = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.norm_layer = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)

    def forward(self, h):
        clust_reps = self.clusters(torch.arange(self.num_clusters))[None, :, :].repeat(h.size(0), 1, 1)
        clust_reps = self.clust_proj(clust_reps)
        h = self.in_proj(h)

        in_emb = clust_reps
        for i in range(self.num_iter):
            clust_reps, weights = self.in_attn(in_emb, h, h)
            # clust_reps = weights @ h
            in_emb = in_emb + clust_reps
            in_emb = self.norm_layer(in_emb.transpose(1, 2)).transpose(1, 2)
        # |weights| = (batch, cluster_number, n_nodes)
        ##（num_cluster,batch,2*batch）
        return None, in_emb, weights


def gather_PAD_AVG(a):
    # 步骤1: 检查每个向量是否全为零
    batch, rollout, k, embeddings = a.size()
    is_zero = a.sum(dim=-1) == 0

    # 步骤2: 计算每个(batch, rollout)组合中非零向量的总和
    non_zero_sum = a.view(batch * rollout, k, embeddings)  # 重塑以简化操作
    # print(non_zero_sum)
    non_zero_mask = ~is_zero.view(batch * rollout, k)  # 对应的掩码，也需要重塑
    # print(non_zero_mask)
    # 使用掩码来只计算非零向量的和
    # 注意：这里我们乘以一个与a形状相同的掩码（扩展为(batch*rollout, k, embeddings)），
    # 但实际上我们只需要在embeddings维度上求和，所以乘以扩展后的掩码（其中最后一维为True/False）
    # 是多余的，因为我们稍后会在dim=2上求和。但为了清晰，我还是保留了这一步，
    # 并用另一种方式来实现（即直接在求和时考虑）。
    # 更简洁的方式是直接求和，然后在后面处理分母。

    # 直接求和，不考虑零向量（因为它们在embeddings维度上的和自动为0）
    total_sum = non_zero_sum.sum(dim=1)  # (batch*rollout, k)
    # print(total_sum)
    # 计算每个(batch, rollout)组合中非零向量的数量
    non_zero_count = non_zero_mask.sum(dim=1, dtype=torch.float32)  # (batch*rollout,)
    # print(non_zero_count)
    # 避免除以零
    non_zero_count = torch.clamp(non_zero_count, min=1e-9)  # 使用一个非常小的数来避免除以零
    # print(non_zero_count)

    # 计算平均值（注意这里我们是在k的维度上“广播”平均值）
    mean_vectors = total_sum / non_zero_count.unsqueeze(-1)
    # print(mean_vectors)
    # 如果需要，将mean_vectors的形状从(batch*rollout, k, 1)调整为(batch*rollout, k, embeddings)
    # 但实际上，由于我们在embeddings维度上进行了平均，所以平均值在embeddings维度上是相同的
    # 这里我们保持其形状为(batch*rollout, k, 1)，以便在后面的步骤中轻松扩展

    # 步骤3: 替换全零向量为平均值向量
    # 首先，将mean_vectors扩展回原始张量的embeddings维度
    mean_vectors_expanded = mean_vectors.unsqueeze(1).expand(batch * rollout, k, embeddings)
    # print(mean_vectors_expanded)
    # 然后，使用torch.where替换全零向量为平均值向量
    # 注意：我们需要将is_zero扩展回原始张量的形状
    b = a.view(batch * rollout, k, embeddings)
    result = torch.where(is_zero.view(batch * rollout, k, 1).expand_as(b), mean_vectors_expanded, b)

    # 最后，将结果重塑回原始形状
    b = result.view(batch, rollout, k, embeddings)

    return b


def gather_PAD_AVG_2(a):
    # 步骤1: 检查每个向量是否全为零
    # print("a", a.size())
    batch = a.size(0)
    rollout = a.size(1)
    k = a.size(2)
    embeddings = a.size(3)

    a = a.reshape(batch * rollout, k, embeddings)
    is_zero = a.sum(dim=-1) == 0  # 对每个向量求和后判断是否为零
    # print(a)
    # 步骤2: 计算非零向量的平均
    non_zero_mask = ~is_zero  # 非零向量的掩码
    # print(non_zero_mask)
    # non_zero_vectors = a[non_zero_mask]  # 提取非零向量
    # # 需要对每个实例单独计算平均
    # print(non_zero_vectors)

    mean_vectors = []

    for i in range(batch * rollout):
        # 找到当前实例中的非零向量
        # print("i=",i)
        # print(non_zero_mask[i, :])
        # current_non_zero = a[(i,(non_zero_mask[i, :]).nonzero(as_tuple=True))]
        non_zero_vectors = a[i, :, :]
        current_non_zero = non_zero_vectors[(non_zero_mask[i, :]).nonzero(as_tuple=True)]
        # print("current",current_non_zero)
        if len(current_non_zero) > 0:
            mean_vector = current_non_zero.mean(dim=0)
        else:
            # 如果当前实例没有非零向量，可以选择保留零向量或者用一个默认值
            mean_vector = torch.zeros(embeddings, dtype=a.dtype, device=a.device)
        mean_vectors.append(mean_vector.expand(k, embeddings))  # 扩展到k个相同的平均值向量
    mean_vectors = torch.stack(mean_vectors)  # 堆叠成(batch, k, embeddings)形状

    # 步骤3: 替换全零向量为平均值向量
    b = torch.where(is_zero.unsqueeze(-1).expand_as(a), mean_vectors, a)
    b = b.reshape(batch, rollout, k, embeddings)
    return b


def search_unvist(a, distance, k, masked=None):
    batch_size, rollout_size = a.shape
    problem_size = distance.shape[1]

    # 初始化结果张量
    c = torch.empty(batch_size, rollout_size, k, dtype=torch.long, device=a.device)
    padding_size = k
    # 遍历每个batch和rollout
    for i in range(batch_size):
        for j in range(rollout_size):
            # 获取当前位置的索引
            current_index = a[i, j]

            # 获取当前位置到其他所有位置的距离（排除自身）
            distances_to_others = distance[i, current_index, :]
            # mask = torch.arange(problem_size, device=a.device) != current_index
            mask_inf = masked[i, j, :]
            # mask1 = mask_inf != -np.inf
            # mask2 = distances_to_others < 0.4
            # mask = mask1 & mask2

            mask = mask_inf != -np.inf

            distances_to_others = distances_to_others[mask]
            # print(distances_to_others)
            # 如果剩余的距离少于k个，则直接取所有剩余的距离
            if distances_to_others.numel() < k:
                k_nearest = distances_to_others.argsort()[:distances_to_others.numel()]

            else:
                # 找到最近的k个索引
                _, k_nearest = torch.topk(distances_to_others, k, largest=False)

                # 将索引映射回原始索引（考虑到之前的mask）
            sorted_indices = torch.arange(problem_size, device=a.device)[mask]
            k_nearest_indices = sorted_indices[k_nearest]
            if len(k_nearest_indices) != k:
                # padding_tensor = torch.zeros(k - k_nearest_indices.size(0), dtype=torch.int32)
                padding_size = k - k_nearest_indices.size(0)
                padding_tensor = torch.full((k - k_nearest_indices.size(0),), fill_value=current_index,
                                            dtype=torch.int32)
                k_nearest_indices = torch.cat((k_nearest_indices, padding_tensor), dim=0)
            # 存储结果
            c[i, j, :] = k_nearest_indices
    return c, padding_size


def search_unvisited(current, distance, k, masked=None):
    batch_size, rollout_size = current.shape
    problem_size = distance.shape[1]
    # print("problem_size",problem_size)

    mask_inf = masked != -np.inf

    # # 确保masked是一个布尔型张量
    # if masked is not None:
    #     mask_inf = masked.to(torch.bool)
    # else:
    #     # 如果没有masked，则默认所有点都是可访问的
    #     mask_inf = torch.ones_like(distance[..., 0:1], dtype=torch.bool)

    # 初始化结果张量
    c = torch.full((batch_size, rollout_size, k), -1, dtype=torch.long, device=current.device)

    # 遍历batch和rollout（实际上是使用索引矩阵来避免显式循环）
    indices = torch.arange(batch_size, device=current.device)[:, None, None] * rollout_size + torch.arange(rollout_size,
                                                                                                           device=current.device)[
                                                                                              None, :, None]

    # print("indices",indices)
    # 获取当前位置索引
    current_indices = current.view(-1)

    # print("current_indices",current_indices)
    # 展开距离矩阵，使其与current_indices的形状兼容
    distance_expanded = distance.repeat(1, rollout_size, 1)
    distance_expanded = distance_expanded.reshape(batch_size * rollout_size, problem_size, problem_size)

    # print("distance_expanded",distance_expanded)
    # 过滤距离矩阵，只保留未被masked的条目
    indices = indices.long()
    current_indices = current_indices.long()

    distance_filtered = distance_expanded[indices.flatten(), current_indices, :]
    # print("distance_filtered",distance_filtered)
    distance_filtered = distance_filtered.reshape(batch_size, rollout_size, problem_size)
    # print("distance_filtered", distance_filtered)
    # print("distance_filtered",distance_filtered)

    # 应用masked掩码

    # 或者更简洁地
    inf_tensor = torch.full_like(distance_filtered, float('inf'))
    # print("inf_tensor",inf_tensor)
    # 使用 torch.where
    distance_masked = torch.where(mask_inf, distance_filtered, inf_tensor)
    # print("distance_masked",distance_masked)
    # print("distance_masked",distance_masked)
    # 找到每个位置最近的k个点
    k_value, k_nearest_indices = torch.topk(distance_masked, k, largest=False, dim=2)
    # print(k_value)
    # print(k_value)
    # 处理不足k个的情况
    # padding_size = torch.clamp(torch.tensor(k - k_nearest_indices.size(2)), min=0)
    # print(padding_size)
    # padding_tensor = torch.full((batch_size, rollout_size, padding_size),
    #                             fill_value=current_indices.view(batch_size, rollout_size, 1).expand(-1, -1,
    #                                                                                                 padding_size),
    #                             dtype=torch.long)
    # padding_tensor = torch.full((batch_size, rollout_size, padding_size),fill_value=0,dtype=torch.long)

    # print("padding_tensor",padding_tensor)
    # k_nearest_indices = torch.cat((k_nearest_indices, padding_tensor), dim=2)
    # print("k_nearest_indices",k_nearest_indices)
    is_inf = k_value == float('inf')
    # print("is_inf",is_inf)
    new_tensor = k_nearest_indices.clone().to(dtype=current.dtype)
    # print("new_tensor",new_tensor)
    expanded_current = torch.full((batch_size, rollout_size, k), problem_size)
    # expanded_current = current[..., None].expand_as(k_value)
    # print(expanded_current[is_inf])
    # print("expanded_current",expanded_current)
    expanded_current = expanded_current.to(dtype=new_tensor.dtype)
    new_tensor[is_inf] = expanded_current[is_inf]
    # print("new_tensor",new_tensor)

    return new_tensor


# 注意：这里假设输入的tensor已经位于适当的设备上（如CUDA），或者为CPU。
# 如果需要，请确保所有输入tensor都已经被移至相同的设备上。


class DiagonalMLP(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, bias=True):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, emb_dim)
        self.layer2 = nn.Linear(emb_dim, out_dim)

    def forward(self, h):
        # h size
        h = self.layer1(h)
        h = torch.relu(h)
        h = self.layer2(h)
        # h = torch.diag_embed(h)

        return h


class UnvisitedMLP(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, bias=True):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, emb_dim)
        self.layer2 = nn.Linear(emb_dim, out_dim)

    def forward(self, h):
        # h size
        h = self.layer1(h)
        h = torch.relu(h)
        h = self.layer2(h)
        # h = torch.diag_embed(h)

        return h


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # print("score",score)
    # shape: (batch, head_num, n, problem)
    # print("score",score.size())
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    # print("score_scaled",score_scaled)
    if rank2_ninf_mask is not None:
        # print("1")
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        # print("2")
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
        # print("score_scaled", score_scaled)

    weights = nn.Softmax(dim=3)(score_scaled)
    # print("weights",weights)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


def fast_multi_head_attention(q, k, v, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :]
        mask = mask.expand(batch_s, head_num, n, input_s)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()

        EMBEDDING_DIM = model_params['embedding_dim']
        HEAD_NUM = model_params['head_num']
        val_dim = EMBEDDING_DIM // HEAD_NUM
        key_dim = val_dim

        self.n_heads = HEAD_NUM
        self.input_dim = EMBEDDING_DIM
        self.embed_dim = EMBEDDING_DIM
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W2_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W3_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))

        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W5_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))
        self.W6_query = nn.Parameter(torch.Tensor(HEAD_NUM, EMBEDDING_DIM, key_dim))

        self.W_out = nn.Parameter(torch.Tensor(HEAD_NUM, key_dim, EMBEDDING_DIM))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: datasets (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        # 在编码器中做的第一次multi head attention中，graph_size = problem+1 ,inpuy_dim = 128
        n_query = q.size(1)
        # n_qu
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]
        # 将h和q展平到两个维度上

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        shp_q = (self.n_heads, batch_size, n_query, -1)
        # 定义两个形状元组，用来reshape

        # pickup -> its delivery attention
        n_pick = (graph_size - 1) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)
        # 定义两个形状元组，用来reshape，交、取货节点

        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # shape[head,batch,graph,key_dim]
        # [batch_size * n_query, embed_dim]*[HEAD_NUM, EMBEDDING_DIM, key_dim]
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        # shape(n_heads, batch_size, graph_size,key_dim)
        V = torch.matmul(hflat, self.W_val).view(shp)
        # shape(n_heads, batch_size, graph_size,key_dim)
        # 如上计算了整体问题的QKV

        # pickup -> its delivery
        pick_flat = h[:, 1:n_pick + 1, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick + 1:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        # 拿出取货和送货节点的嵌入

        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # 取货节点Q和送货节点的k，v

        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick)  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        # 取货节点的QKV

        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # 送货节点的QKV

        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device)
        ], 2)
        # shape [self.n_heads, batch_size, 1 + 2*n_pick, self.input_dim // self.n_heads]，不足的地方用0补足

        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)
        # 交货的q和取货的kv

        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        # 全交货的qkv

        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        # 交货的Q，取货的KV

        # delivery -> its pick up
        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device),
            V_pick  # [n_heads, batch_size, n_pick, key/val_size]
        ], 2)
        # shape [self.n_heads, batch_size, 1 + 2*n_pick, self.input_dim // self.n_heads]，不足的地方用0补足

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        ##Pick up
        # ??pair???attention??
        compatibility_pick_delivery = self.norm_factor * torch.sum(Q_pick * K_delivery,
                                                                   -1)  # element_wise, [n_heads, batch_size, n_pick]
        # [n_heads, batch_size, n_pick, n_pick]
        # 最终，compatibility_pick_delivery 将是一个形状为 (batch_size, n_pick) 的张量，其中每个元素表示批次中对应查询与图中所有节点的相似度之和，并且已经过缩放处理
        compatibility_pick_allpick = self.norm_factor * torch.matmul(Q_pick_allpick, K_allpick.transpose(2,
                                                                                                         3))  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_pick_alldelivery = self.norm_factor * torch.matmul(Q_pick_alldelivery, K_alldelivery.transpose(2,
                                                                                                                     3))  # [n_heads, batch_size, n_pick, n_pick]

        ##Delivery
        compatibility_delivery_pick = self.norm_factor * torch.sum(Q_delivery * K_pick,
                                                                   -1)  # element_wise, [n_heads, batch_size, n_pick]

        compatibility_delivery_alldelivery = self.norm_factor * torch.matmul(Q_delivery_alldelivery,
                                                                             K_alldelivery2.transpose(2,
                                                                                                      3))  # [n_heads, batch_size, n_pick, n_pick]

        compatibility_delivery_allpick = self.norm_factor * torch.matmul(Q_delivery_allpickup, K_allpickup2.transpose(2,
                                                                                                                      3))  # [n_heads, batch_size, n_pick, n_pick]

        ##Pick up->
        # compatibility_additional?pickup????delivery????attention(size 1),1:n_pick+1??attention,depot?delivery??
        compatibility_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            compatibility_pick_delivery,  # [n_heads, batch_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], -1).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_allpick = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_pick_allpick,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_alldelivery = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_pick_alldelivery,  # [n_heads, batch_size, n_pick, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device)
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)
        # [n_heads, batch_size, n_query, graph_size+1+n_pick+n_pick]

        ##Delivery->
        compatibility_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, dtype=compatibility.dtype, device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_pick  # [n_heads, batch_size, n_pick]
        ], -1).view(self.n_heads, batch_size, graph_size, 1)

        compatibility_additional_alldelivery2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_alldelivery  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility_additional_allpick2 = torch.cat([  # [n_heads, batch_size, graph_size, n_pick]
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=compatibility.dtype,
                                 device=compatibility.device),
            compatibility_delivery_allpick  # [n_heads, batch_size, n_pick, n_pick]
        ], 2).view(self.n_heads, batch_size, graph_size, n_pick)

        compatibility = torch.cat([compatibility, compatibility_additional_delivery, compatibility_additional_allpick,
                                   compatibility_additional_alldelivery,
                                   compatibility_additional_pick, compatibility_additional_alldelivery2,
                                   compatibility_additional_allpick2], dim=-1)

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility,
                             dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2] (graph_size include depot)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        # heads: [n_heads, batrch_size, n_query, val_size], attn????pick?deliver?attn
        heads = torch.matmul(attn[:, :, :, :graph_size], V)  # V: (self.n_heads, batch_size, graph_size, val_size)

        # heads??pick -> its delivery
        heads = heads + attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size,
                                                       1) * V_additional_delivery  # V_addi:[n_heads, batch_size, graph_size, key_size]

        # heads??pick -> otherpick, V_allpick: # [n_heads, batch_size, n_pick, key_size]
        # heads: [n_heads, batch_size, graph_size, key_size]
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick),
            V_allpick)

        # V_alldelivery: # (n_heads, batch_size, n_pick, key/val_size)
        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size,
                                                                                    graph_size, n_pick), V_alldelivery)

        # delivery
        heads = heads + attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size,
                                                                        1) * V_additional_pick

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads,
                                                                                                batch_size, graph_size,
                                                                                                n_pick), V_alldelivery2)

        heads = heads + torch.matmul(
            attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick),
            V_allpickup2)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out