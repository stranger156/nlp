import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn

from RNN_module import get_embedding_matrix, RNNconfig, RNNmodel, get_vocab, LabelSmoothingCrossEntropy


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


vocab = get_vocab()
vocab_size = len(vocab)

matrix = get_embedding_matrix('glove', vocab)
print(matrix)
embedding_matrix = torch.Tensor(matrix)

config = RNNconfig(vocab_size)

# 创建一个模型实例
model = RNNmodel(config, embedding_matrix=embedding_matrix)

# 应用权重初始化
model.apply(weights_init)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载处理后的数据集
train_df = pd.read_csv('data/new_train.tsv', sep='\t')
val_df = pd.read_csv('data/new_val.tsv', sep='\t')
test_df = pd.read_csv('data/new_test.tsv', sep='\t')

# 转换为整数格式的输入和标签
train_phrases = [list(map(int, phrase.split())) for phrase in train_df['Phrase']]
train_labels = train_df['Sentiment'].values
val_phrases = [list(map(int, phrase.split())) for phrase in val_df['Phrase']]
val_labels = val_df['Sentiment'].values

test_phrases = [list(map(int, phrase.split())) for phrase in test_df['Phrase']]
test_ids = test_df['PhraseId'].values

# 将数据转换为Tensor
train_phrases_tensor = torch.tensor(train_phrases, dtype=torch.long).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
val_phrases_tensor = torch.tensor(val_phrases, dtype=torch.long).to(device)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(device)

test_phrases_tensor = torch.tensor(test_phrases, dtype=torch.long).to(device)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(train_phrases_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(val_phrases_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_phrases_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
model = model.to(device)  # 将模型移动到GPU
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 训练模型
# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0
        correct = 0
        total = 0

        for phrases, labels in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(phrases)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计正确预测的数量
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')

        # 在每个epoch结束时进行验证
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


# 评估模型
def evaluate_model(model, val_loader, criterion):
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for phrases, labels in val_loader:
            outputs = model(phrases)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return val_loss / len(val_loader), accuracy


def predict(model, test_loader):
    model.eval()
    predictions = []
    data = {}

    with torch.no_grad():
        for phrases in test_loader:
            outputs = model(phrases[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())

    data['PhraseId'] = test_ids
    data['Sentiment'] = predictions
    df = pd.DataFrame(data)
    df.to_csv('predict_result_glove_batch128.csv', index=False)
    return


# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
torch.save(model.state_dict(), 'rnn_model_glove_batch128.pth')

# 加载保存的模型状态字典
model = RNNmodel(config)  # 重新实例化模型
model.load_state_dict(torch.load("rnn_model_glove_batch128.pth"))
model.to(device)
predict(model, test_loader)
