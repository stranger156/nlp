import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
#   导入词汇表
vocab_path = 'data/vocab.txt'
vocab = {}

with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        index, word = line.strip().split('\t')
        vocab[int(index)] = word
vocab_size = len(vocab)
print(vocab)


def get_embedding_matrix(category, vocab):
    find_word = 0
    embedding_dim = 300
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    if category == 'word2vec':
        word2vec_module = api.load("word2vec-google-news-300")
        #   进行嵌入矩阵的填充
        for idx, wrd in vocab.items():
            if wrd in word2vec_module:
                find_word += 1
                embedding_matrix[idx] = word2vec_module[wrd]
            else:
                #   如果不存在该词语，则随机化处理
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    elif category == 'glove':
        glove_module = api.load("glove-wiki-gigaword-300")
        #   进行嵌入矩阵的填充
        for idx, wrd in vocab.items():
            if wrd in glove_module:
                find_word += 1
                embedding_matrix[idx] = glove_module[wrd]
            else:
                #   如果不存在该词语，则随机化处理
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    print(f"找到的单词占{find_word / vocab_size:.2%}")
    return embedding_matrix


matrix = get_embedding_matrix('word2vec', vocab)
print(matrix)
embedding_matrix = torch.Tensor(matrix)


class RNNconfig:
    def __init__(self):
        self.embedding_dim = 300  # 嵌入层维度，因为使用了预训练模型选用的300
        self.hidden_dim = 256  # 隐藏层维度，可以从256开始调整
        self.output_dim = 5  # 输出维度，五个类别所以是5
        self.n_layers = 2  # RNN层数，通常为2
        self.bidirectional = True  # 是否使用双向RNN
        self.dropout = 0.5  # Dropout概率
        self.vocab_size = vocab_size  # 词汇表大小,记得导入vocab然后得到词汇表大小
        self.pad_idx = 0  # <PAD>索引
        self.sos_idx = 2  # <SOS>索引
        self.eos_idx = 3  # <EOS>索引
        self.unk_idx = 1  # <UNK>索引

class RNNmodel(nn.Module):
    def __init__(self, config: RNNconfig, embedding_matrix: torch.Tensor = None):
        super(RNNmodel, self).__init__()

        #  定义嵌入层，如果有嵌入矩阵则使用预训练的嵌入矩阵
        if embedding_matrix is None:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=config.pad_idx)

        #  定义LSTM层
        self.rnn = nn.LSTM(config.embedding_dim,
                           config.hidden_dim,
                           num_layers=config.n_layers,
                           dropout=config.dropout,
                           bidirectional=config.bidirectional,
                           batch_first=True)

        #  定义全连接层，将输出转换为结果
        self.fc = nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim,
                            config.output_dim)

        #  定义dropout层防止过拟合
        self.dropout = nn.Dropout(config.dropout)

        #  归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        # text: [batch_size, sent_len]

        # 嵌入层
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, sent_len, embedding_dim]

        # RNN层
        output, (hidden, cell) = self.rnn(embedded)

        # 如果使用双向RNN，需要将前向和后向的hidden state拼接起来
        if config.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # hidden: [batch_size, hidden_dim * num_directions]

        # 全连接层
        logits = self.fc(hidden)

        #  归一化
        probabilities = self.softmax(logits)

        return probabilities


# 使用示例
config = RNNconfig()

# 创建一个模型实例
model = RNNmodel(config, embedding_matrix=embedding_matrix)

# 模拟一个batch的输入数据
# 假设我们有一个batch_size=4，每个句子长度为10
batch_size = 4
sent_len = 10

# 随机生成一个输入张量
input_tensor = torch.randint(0, config.vocab_size, (batch_size, sent_len))

# 前向传播测试
output = model(input_tensor)

# 检查输出的维度是否正确
assert output.shape == (batch_size, config.output_dim), \
    f"期望得到的维度 {(batch_size, config.output_dim)}, 但是得到了 {output.shape}"

print(output)
print("模型测试通过")







