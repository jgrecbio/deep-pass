from copy import deepcopy
from typing import List, Tuple, Dict, Iterable
from collections import Counter
from toolz import concat

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


def get_label_vectorizer(psswds: List[str],
                         vocabulary_size: int = 2048) -> Tuple[Dict[str, int],
                                                               Dict[int, str]]:
    counter = Counter(concat(psswds))
    label2index = {x[0]: i + 1
                   for i, x
                   in enumerate(counter.most_common(vocabulary_size - 1))}
    label2index["unk"] = 2049
    label2index["pad"] = 0
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
    x = deepcopy(x)
    np.random.shuffle(x)
    for i in range(0, len(x), batch_size):
        yield x[i: i + batch_size]


if __name__ == "__main__":
    psswds = get_data("data/rockyou-withcount.txt")
    l2i, i2l = get_label_vectorizer(psswds)
    print(len(l2i))
    npsswds = vectorizes_psswds(psswds, l2i)
    print(npsswds.shape)
    ohe = get_ohe(npsswds)
    vec_psswds = ohe_vectorizes(ohe, npsswds)
    print(vec_psswds.shape)
