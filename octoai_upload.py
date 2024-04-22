import os
from octoai.clients.asset_orch import AssetOrchestrator, FileData
file_path = '/egr/research-dselab/renjie3/renjie/octoai.txt'

# Read the API key from the file
with open(file_path, 'r') as file:
    api_key = file.read().strip()  # strip() removes any leading/trailing whitespace

os.environ['OCTOAI_TOKEN'] = api_key

id_output_file_path = "octoat_id.txt"

if __name__ == "__main__":
    # OCTOAI_TOKEN set as an environment variable so do not need to pass a token.
    asset_orch = AssetOrchestrator()

    local_dir_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/conceptual_1k_filterwm_templateonly5/"
    # dir_path = "./finetuning_images/"  # Set your dir_path here to your file assets.
    files = []
    # Get a list of files in the folder
    for file_path in os.listdir(local_dir_path):
        if os.path.isfile(os.path.join(local_dir_path, file_path)) and 'jsonl' not in file_path:
            files.append(file_path)
    with open(id_output_file_path, 'w', encoding='utf-8') as output_file:
        for i, file in enumerate(files):
            # Use the file names to get file_format and the asset_name.
            split_file_name = file.split(".")
            asset_name = split_file_name[0]
            file_format = split_file_name[1]
            file_data = FileData(
                file_format=file_format,
            )
            asset = asset_orch.create(
                file=local_dir_path + file,
                data=file_data,
                name=asset_name,
            )

            output_file.write(str(asset).strip() + '\n')

            print(i, str(asset))

# asset_orch = AssetOrchestrator()
# # # List public OctoAI assets
# asset_list = asset_orch.list()
# # print(asset_list)
# # import pdb ; pdb.set_trace()
# for i in range(len(asset_list)):
#     print(i)
#     asset_orch.delete(asset_list[i].id)
# # print(len(asset_orch.list()))
# import pdb ; pdb.set_trace()
# # Get a specific asset, for example, a product photography asset
# asset = asset_orch.get(is_public=True, owner="octoai", name="product_photography_v1")
