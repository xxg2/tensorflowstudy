import tensorflow as tf
import time
import numpy as np
with open('datasets/jay.txt', 'r') as f:
    text = f.read()

#得到了文章中所有的字符集合 vocab
#得到一个字符 - 数字的映射 vocab_to_int
#得到一个数字 - 字符的映射 int_to_vocab
#对原文进行转码后的列表 encoded
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

def get_batches(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[: batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))