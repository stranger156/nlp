import pandas as pd
import numpy as np
from collections import Counter

# 读取数据
train_df = pd.read_csv('data/train.tsv', sep='\t')
test_df = pd.read_csv('data/test.tsv', sep='\t')

# 步骤1: 句子处理
max_length = 2
train_sentences = train_df['Phrase'].dropna().tolist()  # 去掉缺失值
test_sentences = test_df['Phrase'].tolist()

print(len(test_sentences))
# 计算最大长度
for sent in train_sentences:
    if isinstance(sent, str):  # 确保 sent 是字符串
        max_length = max(max_length, len(sent.split()) + 2)


# 填充并添加标记
def process_sentences(sentences):
    processed = []
    for sentence in sentences:
        # 将句子分词
        if not isinstance(sentence, str):
            sentence = str(sentence)
            print(sentence)
        temp_words = sentence.split()
        if len(temp_words) > max_length - 2:
            temp_words = temp_words[:max_length - 2]
        # 添加<SOS>和<EOS>
        temp_words = ['<SOS>'] + temp_words + ['<EOS>']
        # 填充句子
        padding_length = max_length - len(temp_words)
        if padding_length > 0:
            temp_words = temp_words + ['<PAD>'] * padding_length
        processed.append(' '.join(temp_words))
    return processed


train_processed = process_sentences(train_sentences)
test_processed = process_sentences(test_sentences)
# 步骤2: 词表构建
# 使用训练集构建词表
all_words = ' '.join(train_sentences).split()
word_counts = Counter(all_words)

# 只保留出现次数大于0的单词
word_counts = {word: count for word, count in word_counts.items() if count > 0}
words = list(word_counts.keys())

# 添加特殊标记
words = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(words)

# 创建词表字典
word_to_id = {word: idx for idx, word in enumerate(words)}

# 保存词表
with open('data/vocab.txt', 'w', encoding='utf-8') as f:
    for word, idx in word_to_id.items():
        f.write(f"{idx}\t{word}\n")  # 使用制表符分隔 ID 和单词

# 步骤3: 替换为单词ID
def replace_with_ids(processed_sentences):
    id_sentences = []
    for sentence in processed_sentences:
        ids = [str(word_to_id.get(word, word_to_id['<UNK>'])) for word in sentence.split()]
        id_sentences.append(' '.join(ids))
    return id_sentences

train_ids = replace_with_ids(train_processed)
test_ids = replace_with_ids(test_processed)

# 创建新的训练集和测试集
train_phrase_id = train_df['PhraseId'].dropna().reset_index(drop=True)
train_sentiment = train_df['Sentiment'].dropna().reset_index(drop=True)
# 确保 train_ids 的长度与上面的一致
if len(train_phrase_id) != len(train_ids) or len(train_sentiment) != len(train_ids):
    print(f"Length mismatch: PhraseId length {len(train_phrase_id)} != train_ids length {len(train_ids)}")
    # 可能需要做一些处理，比如丢弃多余的
    min_length = min(len(train_phrase_id), len(train_ids), len(train_sentiment))
    train_phrase_id = train_phrase_id[:min_length]
    train_ids = train_ids[:min_length]
    train_sentiment = train_sentiment[:min_length]

new_train_df = pd.DataFrame({
    'PhraseId': train_phrase_id,
    'Phrase': train_ids,
    'Sentiment': train_sentiment,
})

# 对测试集进行相同处理
test_phrase_id = test_df['PhraseId'].dropna().reset_index(drop=True)
print(test_phrase_id)

if len(test_phrase_id) != len(test_ids):
    print(f"Length mismatch: PhraseId length {len(test_phrase_id)} != test_ids length {len(test_ids)}")
    # 处理同样的方式
    min_length = min(len(test_phrase_id), len(test_ids))
    test_phrase_id = test_phrase_id[:min_length]
    test_ids = test_ids[:min_length]

new_test_df = pd.DataFrame({
    'PhraseId': test_phrase_id,
    'Phrase': test_ids
})

# 保存新的数据集，使用制表符作为分隔符
new_train_df.to_csv('data/new_train.tsv', index=False, header=True, sep='\t', encoding='utf-8')
new_test_df.to_csv('data/new_test.tsv', index=False, header=True, sep='\t', encoding='utf-8')

print("处理完成！")