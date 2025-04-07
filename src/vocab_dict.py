import pickle
import torch

from common import token_features_len, vocab_types_len, vocab_to_id_map_filepath, id_to_vocab_map_filepath, id_to_token_map_filepath


vocab_to_id_map = None


def get_vocab_to_id_map():
    global vocab_to_id_map
    if vocab_to_id_map == None:
        with open(vocab_to_id_map_filepath, "rb") as f:
            vocab_to_id_map = pickle.load(f)
    return vocab_to_id_map


id_to_token_map = None


def get_id_to_token_map():
    global id_to_token_map
    if id_to_token_map == None:
        with open(id_to_token_map_filepath, "rb") as f:
            id_to_token_map = pickle.load(f)
    return id_to_token_map


id_to_vocab_map = None


def get_id_to_vocab_map():
    global id_to_vocab_map
    if id_to_vocab_map == None:
        with open(id_to_vocab_map_filepath, "rb") as f:
            id_to_vocab_map = pickle.load(f)
    return id_to_vocab_map


def vocabs_to_ids(vocabs: list, device: str) -> torch.Tensor:
    vocab_ids = torch.empty((len(vocabs),), dtype=torch.int64, device=device)
    for i, vocab in enumerate(vocabs):
        vocab_ids[i] = get_vocab_to_id_map()[vocab]
    return vocab_ids


def vocabs_to_tokens(vocabs: list, device: str) -> torch.Tensor:
    vocab_tokens = torch.empty((len(vocabs), token_features_len), dtype=torch.float32, device=device)
    for i, vocab in enumerate(vocabs):
        vocab_tokens[i] = torch.tensor(get_id_to_token_map()[get_vocab_to_id_map()[vocab]], dtype=torch.float32)
    return vocab_tokens


def id_to_vocab(id: int) -> str:
    return get_id_to_vocab_map()[id]


def id_to_token(id: int, device: str) -> torch.Tensor:
    token = get_id_to_token_map()[id]
    token = torch.tensor(token, dtype=torch.float32, device=device)
    return token


def divide_vocabs(text: str) -> str:
    # todo
    return text


def gen_dict():
    # 读取文本

    text = ""
    with open("./data_src/网文对话.txt", "r", encoding="utf8") as f:
        text += f.read()

    # 分词

    vocabs = divide_vocabs(text)

    # 词->序号映射表

    vocab_types = set(vocabs)
    vocab_to_id_map = {vocab: i for i, vocab in enumerate(vocab_types)}
    with open(vocab_to_id_map_filepath, "wb") as f:
        pickle.dump(vocab_to_id_map, f)

    # 序号->词映射表

    id_to_vocab_map = {i: vocab for i, vocab in enumerate(vocab_types)}
    for i in range(vocab_types_len - len(id_to_vocab_map)):
        id_to_vocab_map[i + len(vocab_types)] = "\0"
    with open(id_to_vocab_map_filepath, "wb") as f:
        pickle.dump(id_to_vocab_map, f)

    # 序号->token映射表（todo：归一化计算）
    id_to_token_map = {}
    for vocab_id, vocab in id_to_vocab_map.items():
        token = []
        for byte in vocab.encode("utf-16be"):
            bits = [(byte >> j) & 1 for j in reversed(range(8))]
            token.extend(bits)
        token = [0] * (token_features_len - len(token)) + token
        id_to_token_map[vocab_id] = token
    with open(id_to_token_map_filepath, "wb") as f:
        pickle.dump(id_to_token_map, f)


if __name__ == "__main__":
    gen_dict()
