# README
[Notes](https://github.com/gatesfoundation/idm-azurestorage-codepsaces-example/blob/main/README.md)
# Setup environment variables 

To avoid storing secrets in the repo, configure your storage account details in your shell as environment variables before running azure-blobfuse-config.yaml  
Following commands use my example blob container, update the values with your own container details.  
Note that blobfuse suppose 4 type of auth(Key, SAS, MSI, SPN), following just demonstrate the use of Key, for more information please see: https://github.com/Azure/azure-storage-fuse/tree/main?tab=readme-ov-file#environment-variables

## For Linux environment (_use this for github codespace environment_):
```bash
export AZURE_STORAGE_BLOB_ENDPOINT="<your_storage_account>.blob.core.windows.net"
export AZURE_STORAGE_ACCOUNT="<your_storage_account>"
export AZURE_STORAGE_ACCESS_KEY="<your_storage_account_key>"
export AZURE_STORAGE_ACCOUNT_CONTAINER="<your_storage_account_container>"
```
Additionally you may also opt to add above lines to your ~/.bashrc so they get setup automatically.