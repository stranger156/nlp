import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader as api
import numpy as np


# 得到词汇表
vocab_path = 'data/vocab.txt'
vocab = {}

with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        index, word = line.strip().split('\t')
        vocab[int(index)] = word
vocab_size = len(vocab)


def get_embedding_matrix(category, vocab):
    find_word = 0
    embedding_dim = 300
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # word embedding随机化
    if category == 'word2vec':
        word2vec_module = api.load("word2vec-google-news-300")
        for idx, wrd in vocab.items():
            if wrd in word2vec_module:
                find_word += 1
                embedding_matrix[idx] = word2vec_module[wrd]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.5, size=(embedding_dim,))
    # glove预训练embedding随机化
    elif category == 'glove':
        glove_module = api.load("glove-wiki-gigaword-300")
        for idx, wrd in vocab.items():
            if wrd in glove_module:
                find_word += 1
                embedding_matrix[idx] = glove_module[wrd]
            else:
                embedding_matrix[idx] = np.random.normal(scale=0.5, size=(embedding_dim,))
    return embedding_matrix


class CNN_config:
    def __init__(self):
        self.seq_length = 54    # 文本预处理后序列的长度
        self.num_classes = 5    # 文本分类的类别数
        self.embedding_dim = 300    # 嵌入层所得到的词向量维度
        self.num_filters = 300  # 经过卷积操作后所得到的特征图数
        self.num_conv1d = 3     # 卷积层的数量
        self.filter_size = [3, 4, 5]    # 设置不同卷积层的卷积核大小
        self.vocab_size = vocab_size    # 指明词汇表大小
        self.line_size = 128    # 设置线性层之间的数据传递的维度
        # 特殊字符的词汇表索引
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3


class CNN_model(nn.Module):
    def __init__(self, config: CNN_config, embedding_matrix: torch.Tensor = None):
        super(CNN_model, self).__init__()
        #  定义嵌入层
        if embedding_matrix is None:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=config.pad_idx)
        # 定义卷积层
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=config.embedding_dim,
                      out_channels=config.num_filters,
                      kernel_size=size)
            for size in config.filter_size
        ])
        self.line1 = nn.Linear(config.num_filters * config.num_conv1d, config.line_size)
        self.line2 = nn.Linear(config.line_size, config.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        inputs = inputs.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        outputs = [
            F.relu(conv1d(inputs)).max(dim=2)[0] for conv1d in self.conv1d_list
        ]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.line1(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        logits = self.line2(outputs)
        probabilities = self.softmax(logits)
        return probabilities


if __name__ == "__main__":
    # 得到嵌入矩阵
    matrix = get_embedding_matrix('word2vec', vocab)
    embedding_matrix = torch.Tensor(matrix)
    # 定义模型参数对象
    config = CNN_config()
    # 定义模型
    cnn_model = CNN_model(config, embedding_matrix=embedding_matrix)
    # 定义输入数据
    Data = [[config.sos_idx, 1946, 1946, 1946, config.eos_idx, config.pad_idx]]
    Tensor_Data = torch.LongTensor(Data)
    # 得到模型预测结果
    result = cnn_model(Tensor_Data)
    print(result)
