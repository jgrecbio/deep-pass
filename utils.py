import boto3 as boto


def get_client():
    client = boto.client('s3')
    return client


def get_dataset(client, bucket, key):
    client.download(bucket, key, key)
