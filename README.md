# MyGPT

GPT-1微缩模型例程，全参数缩小4倍，共缩小1024倍。

## 玩法

安装环境：

```sh
python -m venv venv  # 创建虚拟环境
source venv/script/active  # 当前终端内激活虚拟环境
pip install -r requirements.txt  # 安装依赖包
```

启动：
1. 创建词典：`python src/vocab_dict.py`，将在dict目录下生成词典文件，包含每个词的编解码信息。直接通过数据进行生成，且暂未实现分词。
2. 创建数据：`python src/data.py`，将在data_compiled目录下生成数据文件。
3. 训练模型：`python src/train.py`，将在weights目录下生成模型权重文件。训练所需总显存约30G。
4. 启动命令行交互对话：`python src/play.py`。

## 原理

### 模型输入输出

模型输入数据：三维，样本维（各样本）x元素维（文本序列的各词语）x特征维（词语token的各特征）。
模型输出数据：三维，样本维（各样本）x元素维（文本序列的各词语）x可能性维（各词是下一个词的可能性，结合softmax可化为概率）。
样本维、元素维每次输入长度可变。

### 模型结构

若干个Transformer解码器串联，最后接单层MLP。详见`src/MyGPT.py`。

Transformer解码器：输入输出均为样本维x元素维x特征维。单个Transformer解码器又分为自注意力结构串联MLP结构。
单层MLP：输入为样本维x元素维x特征维，输出为样本维x元素维x可能性维。将特征维转为可能性维。

Transformer解码器-自注意力结构：输入输出均为样本维x元素维x特征维。使用多头自注意力。详见`doc/Attention.png`。
Transformer解码器-MLP结构：输入输出均为样本维x元素维x特征维。
