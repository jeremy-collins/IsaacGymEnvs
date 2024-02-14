import os

token = os.environ.get("SAPIEN_API_TOKEN")

import sapien
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=int, default=101463)

    args = parser.parse_args()
    # 101463 -> spray_bottle
    urdf_file = sapien.asset.download_partnet_mobility(args.object_id, token)
