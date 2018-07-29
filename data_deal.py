import sys
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    sys.setdefauleencoding('utf-8')
    is_py3 = False


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def open_file(filename, mode='r'):
    # mode代表是r读和w写
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    # 依据训练集简历词汇表
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    # 添加一个<UNK>将词汇表长度定格在vocab_size
    words = ['UNK']+list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words)+'\n')


def read_category():
    # 读取分类目录  自己设定
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def read_vocab(vocab_dir):
    # 读取词汇表
    with open_file(vocab_dir) as f2:
        words = [native_content(_.strip()) for _ in f2.readlines()]
    word_to_id = dict(zip(words,range(len(words))))
    return words, word_to_id


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad
