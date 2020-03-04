import json
import argparse

from tensorflow.keras.models import load_model

from generate_passwords import generate_random_psswds, generate_eager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder")
    parser.add_argument("--model")
    parser.add_argument("--password-length", type=int, default=14)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--eager", action="store_true", default=False)

    args = parser.parse_args()

    with open(args.encoder) as f:
        l2i, i2l = json.load(f)
    model = load_model(args.model)
    if args.eager:
        print(generate_eager(model,
                             l2i, i2l,
                             args.password_length))
    else:
        for i in range(args.num):
            print(generate_random_psswds(model,
                                         l2i, i2l,
                                         args.password_length,
                                         temperature=args.temperature))
