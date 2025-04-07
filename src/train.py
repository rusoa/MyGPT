import torch

from MyGPT import MyGPT
from common import tokens_len_max, token_features_len, vocab_types_len, data_filepath, weights_filepath, model_device


batch_samples_len = 64
data_workers_len = 4
epochs_len = 100


class LossEvaluator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        # pred: (samples, vocab_tokens, vocab_types)
        # real: (samples, vocab_ids)

        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        pred = pred.permute(0, 2, 1)
        return torch.nn.functional.nll_loss(pred, real)


def main():
    # 模型初始化

    model = MyGPT(3, 3, tokens_len_max, token_features_len, vocab_types_len).to(model_device)
    try:
        model.load_state_dict(torch.load(weights_filepath))  # 加载模型动参
    except Exception:
        print("default weights")
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    # 准备数据集

    samples = torch.load(data_filepath)
    train_samples_x = samples["train_samples_x"]
    train_samples_y = samples["train_samples_y"]
    validation_samples_x = samples["validation_samples_x"].to(model_device)
    validation_samples_y = samples["validation_samples_y"].to(model_device)
    print(train_samples_x.shape, train_samples_y.shape, validation_samples_x.shape, validation_samples_y.shape)

    # 模型训练

    data_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train_samples_x, train_samples_y),
        batch_size=batch_samples_len,
        shuffle=True,  # 是否打乱数据
        num_workers=data_workers_len,
    )

    loss_evaluator = LossEvaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # lr: learning rate

    model.eval()  # 将模型切换至推理模式，关闭批归一化层和丢弃层功能
    y_pred = model(validation_samples_x)
    loss_validation = loss_evaluator(y_pred, validation_samples_y)
    print(f"loss on validation set: {loss_validation.item()}")
    for epoch_idx in range(epochs_len):
        print(f"epoch: {epoch_idx}/{epochs_len}")
        model.train()  # 将模型切换至训练模式，开启批归一化层和丢弃层功能
        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(model_device)
            y = y.to(model_device)
            # 前向传播
            y_pred = model(x)  # 输入多维数据时，模型将默认第一维为 batch 维
            # 反向传播
            loss = loss_evaluator(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        y_pred = model(validation_samples_x)
        loss_validation_ = loss_evaluator(y_pred, validation_samples_y)
        print(f"loss on validation set: {loss_validation_.item()}")
        if loss_validation_ < loss_validation:
            print(f"saved")
            torch.save(model.state_dict(), weights_filepath)  # 保存模型动参
            loss_validation = loss_validation_


if __name__ == "__main__":
    main()
