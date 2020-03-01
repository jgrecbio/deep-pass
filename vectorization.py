from typing import List, Tuple, Dict, Iterable
from collections import Counter
from toolz import concat

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, label_binarize


def get_label_vectorizer(psswds: List[str],
                         start_end: bool = False,
                         vocabulary_size: int = 2048) -> Tuple[Dict[str, int],
                                                               Dict[int, str]]:
    if start_end:
        st = 2
    else:
        st = 1
    counter = Counter(concat(psswds))
    label2index = {x[0]: i
                   for i, x
                   in enumerate(counter.most_common(vocabulary_size + st), st)}
    label2index["pad"] = 0
    if len(counter.most_common(vocabulary_size)) > vocabulary_size:
        label2index["unk"] = vocabulary_size + st
    if start_end:
        label2index["<start>"] = 1
        label2index["<end>"] = len(label2index)
    index2label = {i: label for (label, i) in label2index.items()}
    return label2index, index2label


def inv_transform(vec_psswds: np.ndarray, i2l: Dict[int, str]) -> List[str]:
    psswds = []
    for i in range(vec_psswds.shape[0]):
        psswds.append(''.join(list(map(lambda x: i2l[x], vec_psswds[i]))))
    return psswds


def vectorizes_psswds(psswds: List[str],
                      label2index: Dict[str, int],
                      maxlen: int = 14) -> np.ndarray:
    l2i = label2index
    vectorized_psswds = []
    for psswd in psswds:
        vectorized_psswds.append(
            list(map(lambda x: l2i.get(x, l2i["unk"]), psswd))
        )
    return pad_sequences(vectorized_psswds,
                         maxlen=maxlen,
                         padding="post",
                         truncating="post",
                         value=l2i.get("pad", 0))


def get_ohe(vec_psswds: np.ndarray) -> Tuple[OneHotEncoder, np.ndarray]:
    r, c = vec_psswds.shape
    ohe = OneHotEncoder(sparse=False)
    flat_psswds = vec_psswds.reshape((-1, 1))
    ohe.fit(flat_psswds)
    return ohe


def get_ohe_rnn(vec_psswds: List[List[int]]) -> OneHotEncoder:
    flat_psswds = np.array(list(concat(vec_psswds)))
    ohe = LabelBinarizer()
    ohe.fit(np.reshape(flat_psswds, (-1, 1)))
    return ohe


def ohe_vectorizes_rnn(l2i: Dict[str, int],
                       vec_psswds: List[List[int]]) -> List[np.ndarray]:
    classes = list(l2i.values())
    res = []
    for label in vec_psswds:
        res.append(label_binarize(label, classes))
    return res


def get_generator_rnn(l2i: Dict[str, int],
                      x: List[np.ndarray],
                      y: List[np.ndarray],
                      max_len: int,
                      batch_size: int) -> Iterable[Tuple[np.ndarray,
                                                         np.ndarray]]:
    while True:
        x_gen = construct_batches(x, batch_size)
        y_gen = construct_batches(x, batch_size)
        gen = zip(x_gen, y_gen)
        for (bx, by) in gen:
            pbx = pad_sequences(bx,
                                maxlen=max_len,
                                truncating="post",
                                padding="post",
                                value=0.)
            pby = pad_sequences(ohe_vectorizes_rnn(l2i, by),
                                maxlen=max_len,
                                truncating="post",
                                padding="post",
                                value=0.)
            yield pbx, pby


def ohe_vectorizes(ohe: OneHotEncoder,
                   vec_psswds: np.ndarray) -> np.ndarray:
    r, c = vec_psswds.shape
    flat_psswds = vec_psswds.reshape((-1, 1))
    return ohe.transform(flat_psswds).reshape((r, c, -1))


def get_data(path: str) -> List[str]:
    psswds = []
    with open(path, encoding="ISO-8859-1") as f:
        for line in f.readlines():
            psswds.append(line.split(' ')[-1].strip())
    return psswds


def construct_batches(x: np.ndarray,
                      batch_size: int = 128) -> Iterable[np.ndarray]:
    np.random.shuffle(x)
    for i in range(0, len(x), batch_size):
        yield x[i: i + batch_size]


def vectorizes_sequences(psswds: Iterable[str],
                         l2i: Dict[str, int]) -> Tuple[List[List[int]],
                                                       List[List[int]]]:
    features = []
    labels = []
    for psswd in psswds:
        curr_psswd = []
        for char in psswd:
            curr_psswd.append(l2i.get(char, l2i.get("unk", 0)))
        features.append([l2i["<start>"]] + curr_psswd)
        labels.append(curr_psswd + [l2i["<end>"]])
    return features, labels


if __name__ == "__main__":
    psswds = get_data("data/rockyou-withcount.txt")
    l2i, i2l = get_label_vectorizer(psswds)
    print(len(l2i))
    npsswds = vectorizes_psswds(psswds, l2i)
    print(npsswds.shape)
    ohe = get_ohe(npsswds)
    vec_psswds = ohe_vectorizes(ohe, npsswds)
    print(vec_psswds.shape)
