import torch

from MyGPT import MyGPT
from common import tokens_len_max, token_features_len, vocab_types_len, weights_filepath, model_device
from vocab_dict import vocabs_to_tokens, id_to_vocab, id_to_token, divide_vocabs


def generate(model: torch.nn.Module, hint: str, temprature: float, end_vocab: set, max_len: int):
    model.eval()  # 将模型切换至推理模式，关闭批归一化层和丢弃层功能
    device = next(model.parameters()).device

    vocabs = divide_vocabs(hint)

    if len(hint) == 0:
        return

    try:
        tokens = vocabs_to_tokens(vocabs, device)
    except KeyError as e:
        print(f"输入中含有非法字符{e}")
        return

    x = tokens.reshape((1, *tokens.shape))  # 添加samples维度
    total_len = 0
    while True:
        y = model(x)
        y = y[0]  # 去掉samples维度
        y = y[-1]  # 取最后一个元素
        if temprature == 0:  # temprature 多样性参数，越高输出越抽象
            vocab_id = torch.argmax(y, dim=-1).item()
        else:
            y = torch.nn.functional.softmax(y / temprature, dim=-1)
            vocab_id = torch.multinomial(y, 1).item()
        vocab = id_to_vocab(vocab_id)
        print(vocab, end="")
        total_len += 1
        if vocab in end_vocab or total_len == max_len:
            print()
            break
        vocab_token = id_to_token(vocab_id, device)
        vocab_token = vocab_token.reshape((1, 1, *vocab_token.shape))
        x = torch.cat((x, vocab_token), dim=1)


def main():
    # 模型初始化

    model = MyGPT(3, 3, tokens_len_max, token_features_len, vocab_types_len).to(model_device)  # GPT-2缩小4倍
    try:
        model.load_state_dict(torch.load(weights_filepath))  # 加载模型动参
    except Exception:
        print("default weights")
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    # 循环输入

    print("请输入文本（按 Ctrl+c 退出）：")
    while True:
        try:
            text = input(">>> ")
        except KeyboardInterrupt:
            exit()
        generate(model, text, 0.5, set(["。", "\n"]), 256)


if __name__ == "__main__":
    main()
