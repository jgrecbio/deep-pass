from typing import List, Dict
import tensorflow as tf
from tensorflow.keras.models import Sequential


def generate_random_psswds(model: Sequential,
                           l2i: Dict[str, int],
                           i2l: Dict[int, str],
                           psswd_length: int,
                           temperature: float) -> List[str]:
    input_eval = ["<start>"]
    for i in range(psswd_length):
        vec_input_eval = tf.convert_to_tensor([[l2i[x] for x in input_eval]])
        predictions = model(vec_input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,
                                             num_samples=1)[-1, 0].numpy()
        input_eval.append(i2l[str(predicted_id)])
    return "".join(input_eval)


def generate_eager(model: Sequential,
                   l2i: Dict[str, int],
                   i2l: Dict[int, str],
                   psswd_length: int) -> str:
    input_eval = ["<start>"]
    for i in range(psswd_length):
        vec_input_eval = tf.convert_to_tensor([[l2i[x] for x in input_eval]])
        predictions = model(vec_input_eval)
        predictions = tf.squeeze(predictions, 0)
        pred_id = tf.argmax(predictions, axis=-1)[-1].numpy()
        input_eval.append(i2l[str(pred_id)])
    return "".join(input_eval)
