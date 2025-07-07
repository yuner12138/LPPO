import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PDTSPCollect(nn.Module):

    def __init__(self, model_params,
                 trainer_params):
        super().__init__()
        self.model_params = model_params
        self.force_first_move = model_params['force_first_move']

        self.encoder = PDTSP_Encoder(**model_params)
        self.decoder = PDTSP_Decoder(**model_params)
        self.encoded_nodes = None
        self.encoded_graph = None
        # shape: (batch, problem, EMBEDDING_DIM)

        self.start_q1 = nn.Parameter(torch.zeros(model_params['embedding_dim']), requires_grad=True)
        self.start_last_node = nn.Parameter(torch.zeros(model_params['embedding_dim']), requires_grad=True)

    def pre_forward(self, reset_state, z):
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.encoded_graph = self.encoded_nodes.mean(dim=1,keepdim=True)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes, z)


    def get_expand_prob(self, state):

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        # shape: (batch, beam_width, embedding)
        probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
        # shape: (batch, beam_width, problem)

        return probs
    #强制从不同起点开始
    def forward(self, state, greedy_construction=False, EAS_incumbent_action=None):
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)
        problem_size = self.encoded_nodes.size(1)

        if self.force_first_move and state.current_node is None:
            selected = torch.arange(problem_size).repeat(rollout_size // problem_size)[None, :].expand(batch_size,
                                                                                                    rollout_size)
            prob = torch.ones(size=(batch_size, rollout_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:#全0开始，随机一个占位符去选择第一个点
            if state.current_node is None:
                self.decoder.set_q1(self.start_q1[None, None].expand(batch_size, rollout_size, -1))
                encoded_last_node = self.start_last_node[None, None].expand(batch_size, rollout_size, -1)
            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)

            # shape: (batch, rollout, embedding)
            probs = self.decoder(self.encoded_graph, encoded_last_node, ninf_mask=state.ninf_mask)
            #print('max_prob={}'.format(probs.max()))
            # shape: (batch, rollout, problem+1)

            if not greedy_construction:
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * rollout_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, rollout_size)
                    # shape: (batch, rollout)

                    if EAS_incumbent_action is not None:
                        selected[:, -1] = EAS_incumbent_action

                    prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
                    #print('prob={}'.format(prob))
                    #print(probs)
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
        #print(selected)
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
        self.init_embed_pick = nn.Linear(4, embedding_dim)
        self.init_embed_delivery = nn.Linear(2, embedding_dim)

        # self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])


    def forward(self, data):
        # datasets.shape: (batch, problem, 2)
        problem = data.size(1)
        # datasets.shape: (batch, problem, 2)
        depot_xy = data[:, [0], :]  # 取出仓库节点位置
        # shape = (batch, 1, 2)
        node_xy = data[:, 1:, :]  # 后续节点位置
        # shape = (batch, problem, 2)

        embed_depot = self.embedding_depot(depot_xy)  # 得到仓库的嵌入

        # [batch_size, graph_size//2, 4] loc
        feature_pick = torch.cat([node_xy[:, :(problem - 1) // 2, :], node_xy[:, (problem - 1) // 2:, :]], -1)
        # 将node_xy沿第二个维度上分为两个部分，再在第三个维度上将这两个部分拼接
        # [batch_size, graph_size//2, 2] loc
        feature_delivery = node_xy[:, (problem - 1) // 2:, :]  # [batch_size, graph_size//2, 2]
        embed_pick = self.init_embed_pick(feature_pick)
        embed_delivery = self.init_embed_delivery(feature_delivery)

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

        #out_concat = self.attention_fn(q, k, v)
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
        lppo_embedding_dim = self.model_params['lppo_embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        z_dim = model_params['z_dim']
       # self.rollouts_size = trainer_params['train_z_sample_size']





        self.use_EAS_layers = False

        self.Wq = nn.Linear(2 * embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

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
        init.zeros_(self.lppo_layer_1.weight)
        init.zeros_(self.lppo_layer_2.weight)

        # 注意：如果你也想初始化偏置为0，可以这样做
        init.zeros_(self.lppo_layer_1.bias)
        init.zeros_(self.lppo_layer_2.bias)
        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention


    def set_kv(self, encoded_nodes, z):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

        self.z = z
        # shape: (batch, rollout, z_dim)

    def set_z(self, z):
        self.z = z

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

    def forward(self, all_embedding,encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        input_cat = torch.cat((all_embedding.expand(-1, encoded_last_node.size(dim=1), -1), encoded_last_node), dim=2)
        #  Multi-Head Attention
        #######################################################
        #q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = reshape_by_heads(self.Wq(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)


        out_concat = self.attention_fn(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, rollout, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)

        # shape: (batch, rollout, embedding)

        if not self.use_EAS_layers:
            lppo_out = self.lppo_layer_1(torch.cat((mh_atten_out, self.z), dim=2))
            # shape: ?
            lppo_out = F.relu(lppo_out)
            # shape: ?
            lppo_out = self.lppo_layer_2(lppo_out)
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


        mh_atten_out += lppo_out

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, rollout, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, rollout, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, rollout, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

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
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
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
        #在编码器中做的第一次multi head attention中，graph_size = problem+1 ,inpuy_dim = 128
        n_query = q.size(1)
        #n_qu
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]
        #将h和q展平到两个维度上

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        #定义两个形状元组，用来reshape

        # pickup -> its delivery attention
        n_pick = (graph_size - 1) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)
        #定义两个形状元组，用来reshape，交、取货节点

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
        #如上计算了整体问题的QKV


        # pickup -> its delivery
        pick_flat = h[:, 1:n_pick + 1, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick + 1:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        #拿出取货和送货节点的嵌入

        # pickup -> its delivery attention
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)
        #取货节点Q和送货节点的k，v

        # pickup -> all pickups attention
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick)  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        #取货节点的QKV

        # pickup -> all delivery
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        #送货节点的QKV

        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device)
        ], 2)
        #shape [self.n_heads, batch_size, 1 + 2*n_pick, self.input_dim // self.n_heads]，不足的地方用0补足

        # delivery -> its pickup attention
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)
        #交货的q和取货的kv

        # delivery -> all delivery attention
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        #全交货的qkv

        # delivery -> all pickup
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)
        #交货的Q，取货的KV

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
        #最终，compatibility_pick_delivery 将是一个形状为 (batch_size, n_pick) 的张量，其中每个元素表示批次中对应查询与图中所有节点的相似度之和，并且已经过缩放处理
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