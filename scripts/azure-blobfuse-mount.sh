#!/bin/bash
set -euo pipefail
set -o errexit
set -o errtrace
IFS=$'\n\t'

# Default mount path
DEFAULT_MOUNT_PATH=/root/azurestorage

# Use the first argument as MOUNT_PATH if provided, otherwise use the default
MOUNT_PATH=${1:-$DEFAULT_MOUNT_PATH}

# Configuring temporary path for caching or streaming
# finally we create an empty directory for mounting the blob container
mkdir /tmp/blobfuse \
    && chown root /tmp/blobfuse \
    && mkdir -p $MOUNT_PATH

# Authorize access to your storage account and mount our blobstore
# Example: https://github.com/Azure/azure-storage-fuse/blob/main/sampleFileCacheConfig.yaml
# Full Config: https://github.com/Azure/azure-storage-fuse/blob/main/setup/baseConfig.yaml
blobfuse2 mount $MOUNT_PATH --config-file=./azure-blobfuse-config.yaml