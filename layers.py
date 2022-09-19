import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0
        assert d_v == int(d_model / n_head)
        assert d_k == int(d_model / n_head)

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scaled = math.sqrt(d_k)

        self.linear_q = nn.Linear(d_model, n_head * d_k)
        self.linear_k = nn.Linear(d_model, n_head * d_k)
        self.linear_v = nn.Linear(d_model, n_head * d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # 鉴于线性层没有有益影响，所以去掉线性层
        # self.output_linear = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        def shape(x):
            return x.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)

        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)

        query = shape(query)
        key = shape(key)
        value = shape(value)

        scores = torch.matmul(query, key.transpose(2, 3)).div(self.scaled)
        # print('scores.shape', scores.shape)
        # print('mask.shape', mask.shape)

        attns = self.dropout(self.softmax(scores))
        context = unshape(torch.matmul(attns, value))

        # output = self.output_linear(context)

        norm_output = self.layernorm(context + v)
        # norm_output = self.layernorm(output + v)

        return norm_output, attns


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # initialization
        # init.xavier_normal_(self.fc1.weight.data)
        # init.xavier_normal_(self.fc2.weight.data)
        # init.kaiming_normal_(self.fc1.weight.data)
        # init.kaiming_normal_(self.fc2.weight.data)

    def forward(self, inputs):
        relu_output = self.dropout1(self.relu(self.fc1(inputs)))
        ffn_output = self.fc2(relu_output)
        output = self.dropout2(self.layernorm(inputs + ffn_output))
        return output