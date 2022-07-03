import os

token = os.environ.get("SAPIEN_TOKEN")

import sapien
# 101463 -> spray_bottle
urdf_file = sapien.asset.download_partnet_mobility(101463, token)

