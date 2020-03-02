from typing import List, Dict
import tensorflow as tf
from tensorflow.keras.models import Sequential


def generate_random_psswds(model: Sequential,
                           l2i: Dict[str, int],
                           i2l: Dict[int, str],
                           psswd_length: int,
                           batch_size: int,
                           depth: int,
                           temperature: float) -> List[str]:
    input_eval = ["<start>"]
    for i in range(psswd_length):
        vec_input_eval = tf.convert_to_tensor([l2i[x] for x in input_eval])
        predictions = model(vec_input_eval.reshape((1, -1)))
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,
                                             num_samples=1)[-1, 0].numpy()
        input_eval.append(i2l[int(predicted_id)])
    return "".join(input_eval)
