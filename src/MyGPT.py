import math

import torch


class MyGPT(torch.nn.Module):
    def __init__(self, processors_len, processor_heads_len, tokens_len_max, token_features_len, token_types_len):
        super().__init__()
        self.tokens_len_max = tokens_len_max
        self.processors = SerialReplicator(lambda: Processor(processor_heads_len, token_features_len, 0.1), processors_len)
        self.classifier = torch.nn.Linear(token_features_len, token_types_len)

    # 输入：样本维x元素维x特征维
    # 输出：样本维x元素维x可能性维
    def forward(self, x):
        x = self.processors(lambda module, x_: module(x_, self.tokens_len_max * 0.45), x)
        x = self.classifier(x)
        return x


class SerialReplicator(torch.nn.Module):
    def __init__(self, make_module, n):
        super().__init__()
        self.modules_list = torch.nn.ModuleList([make_module() for _ in range(n)])

    def forward(self, call_module, x):
        for module in self.modules_list:
            x = call_module(module, x)
        return x


class Processor(torch.nn.Module):
    def __init__(self, heads_len, token_features_len, dropout):
        super().__init__()
        self.self_attention = Smoother(MultiHeadsAttention(heads_len, token_features_len, token_features_len, dropout), token_features_len, dropout)
        self.post_mlp = Smoother(PostMLP(token_features_len, dropout), token_features_len, dropout)

    def forward(self, tokens, rotate_base):
        tokens = self.self_attention(lambda module, x: module(x, x, rotate_base, trib((tokens.shape[-2], tokens.shape[-2]), 1, tokens.device)), tokens)
        tokens = self.post_mlp(lambda module, x: module(x), tokens)
        return tokens


def trib(shape, diagonal, device):
    msk = torch.ones(shape, dtype=torch.uint8, device=device)
    msk = msk - torch.triu(msk, diagonal=diagonal)
    return msk


# 据说有助于梯度传递
# 调用模型 -> dropout -> 与原数据相加 -> 对特征维做标准化
class Smoother(torch.nn.Module):
    def __init__(self, module, features_len, dropout):
        super().__init__()
        self.module = module
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = Norm(features_len)

    def forward(self, call, x):
        x1 = call(self.module, x)
        x1 = self.dropout(x1)
        x = x + x1
        x = self.norm(x)
        return x


# 对最后一维做标准化
class Norm(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(size))
        self.b = torch.nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True) + 1e-6  # 防止 std = 0
        x = self.a * (x - mean) / std + self.b
        return x


class MultiHeadsAttention(torch.nn.Module):
    def __init__(self, heads_len, subject_token_features_len, object_token_features_len, dropout):
        super().__init__()
        self.heads_len = heads_len

        if subject_token_features_len % heads_len != 0:
            raise Exception("subject_token_features_len % heads_len != 0")
        k_features_len = int(subject_token_features_len / heads_len)

        if object_token_features_len % heads_len != 0:
            raise Exception("object_token_features_len % heads_len != 0")
        v_features_len = int(object_token_features_len / heads_len)

        self.q_linear = torch.nn.Linear(subject_token_features_len, heads_len * k_features_len)
        self.k_linear = torch.nn.Linear(object_token_features_len, heads_len * k_features_len)
        self.v_linear = torch.nn.Linear(object_token_features_len, heads_len * v_features_len)
        self.dropout = torch.nn.Dropout(dropout)
        self.c_linear = torch.nn.Linear(heads_len * v_features_len, object_token_features_len)

    # 输出 (subject_tokens, object_token_features)
    def forward(self, subject_tokens, object_tokens, rotate_base, msk):
        q = self.q_linear(subject_tokens)  # 样本x主体序列元素x多头键特征
        q = q.reshape((*q.shape[:-1], self.heads_len, -1))  # 样本x主体序列元素x头x键特征
        q = q.transpose(-3, -2)  # 样本x头x主体序列元素x键特征

        k = self.k_linear(object_tokens)
        k = k.reshape((*k.shape[:-1], self.heads_len, -1))
        k = k.transpose(-3, -2)

        v = self.v_linear(object_tokens)
        v = v.reshape((*v.shape[:-1], self.heads_len, -1))
        v = v.transpose(-3, -2)

        c = self.do_attention(q, k, v, rotate_base, msk)

        c = c.transpose(-3, -2)  # 样本x头x主体序列元素x值特征 -> 样本x主体序列元素x头x值特征
        c = c.flatten(-2)  # 样本x主体序列元素x多头值特征
        c = self.c_linear(c)  # 样本x主体序列元素x客体序列特征

        return c

    # 输出 (subject_tokens, heads * v_features)
    def do_attention(self, q, k, v, rotate_base, msk):
        q = rotate_pos_encode(q, rotate_base)
        k = rotate_pos_encode(k, rotate_base)

        a = q.matmul(k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # 乘并归一化
        if msk is not None:
            a = a.masked_fill(msk == 0, -1e9)
        a = torch.softmax(a, -1)
        a = self.dropout(a)

        c = a.matmul(v)

        return c


# 位置编码，引导元素关注与自身临近的元素
def rotate_pos_encode(x, rotate_base):
    x_features_len = x.size(-1)
    x_seq_len = x.size(-2)

    freqs_len = (x_features_len + 1) // 2
    f = 1.0 / rotate_base ** (torch.arange(0, freqs_len, device=x.device) / freqs_len)
    r = torch.arange(x_seq_len, device=x.device)
    r = torch.outer(r, f)
    r = torch.polar(torch.ones_like(r), r)

    if x_features_len % 2 == 1:  # 奇数特征补一列 0
        x = torch.concat([x, torch.zeros((*x.shape[:-1], 1))], dim=-1)
    x = x.reshape((*x.shape[:-1], -1, 2))
    x = torch.view_as_complex(x)
    x = x * r
    x = torch.view_as_real(x)
    x = x.flatten(-2)
    if x_features_len % 2 == 1:  # 去掉补充的列
        x = torch.slice_copy(x, -1, 0, -1)

    return x


class PostMLP(torch.nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(size, size * 4)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin2 = torch.nn.Linear(size * 4, size)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x
