import ray


if __name__ == '__main__':

    ray.init(address="10.1.1.2:6379", _redis_password="tf_cluster_123")
    total_gpus = ray.cluster_resources()["GPU"]
    avail_gpus = ray.available_resources()["GPU"]

    print(f"{avail_gpus}/{total_gpus} GPUs available")
