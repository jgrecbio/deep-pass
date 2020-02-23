import argparse
import boto3 as boto


def get_client():
    client = boto.client('s3')
    return client


def get_dataset(client, bucket, key):
    client.download_file(bucket, key, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bucket")
    parser.add_argument("-k", "--key")

    args = parser.parse_args()

    client = get_client()
    get_dataset(client, args.bucket, args.key)
