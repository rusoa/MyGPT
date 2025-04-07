import torch

from common import tokens_len_max, token_features_len, data_filepath
from vocab_dict import vocabs_to_ids, vocabs_to_tokens, divide_vocabs


samples_len = 2000
sample_step = 400


def main():
    # 读取文本

    text = ""
    with open("./data_src/网文对话.txt", "r", encoding="utf8") as f:
        text += f.read()

    # 分词

    vocabs = divide_vocabs(text)

    # 序号化、token化

    vocab_ids = vocabs_to_ids(vocabs, torch.get_default_device())
    vocab_tokens = vocabs_to_tokens(vocabs, torch.get_default_device())

    # 生成样本集

    samples_x = torch.empty((samples_len, tokens_len_max, token_features_len), dtype=torch.float32)
    samples_y = torch.empty((samples_len, tokens_len_max), dtype=torch.int64)
    for i in range(samples_len):
        samples_x[i] = vocab_tokens[i * sample_step : i * sample_step + tokens_len_max]
        samples_y[i] = vocab_ids[i * sample_step + 1 : i * sample_step + tokens_len_max + 1]

    # 打乱并划分训练、测试样本集

    shuffle_idxs = torch.randperm(samples_len)
    samples_x = samples_x[shuffle_idxs]
    samples_y = samples_y[shuffle_idxs]
    train_samples_len = int(samples_len * 0.8)
    train_samples_x, train_samples_y = samples_x[:train_samples_len], samples_y[:train_samples_len]
    validation_samples_x, validation_samples_y = samples_x[train_samples_len:], samples_y[train_samples_len:]
    print(train_samples_x.shape, train_samples_y.shape, validation_samples_x.shape, validation_samples_y.shape)

    torch.save(
        {
            "train_samples_x": train_samples_x,
            "train_samples_y": train_samples_y,
            "validation_samples_x": validation_samples_x,
            "validation_samples_y": validation_samples_y,
        },
        data_filepath,
    )


if __name__ == "__main__":
    main()
