from octoai.client import Client
import json
import os
from octoai.clients.asset_orch import AssetOrchestrator
from tqdm import tqdm

file_path = '/egr/research-dselab/renjie3/renjie/octoai.txt'

# Read the API key from the file
with open(file_path, 'r') as file:
    api_key = file.read().strip()  # strip() removes any leading/trailing whitespace

os.environ['OCTOAI_TOKEN'] = api_key

# client = Client()
# tune = client.tune.get("tune_01hvqrr7nxejg9rqwct8d94b6w")

# import pdb ; pdb.set_trace()

if __name__ == "__main__":
    client = Client()
    asset_orch = AssetOrchestrator()

    result_dict = {}
    jsonl_file_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-1k-temonly5.jsonl"
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            result_dict[data['file_name'].replace(".png", "")] = "sks1 " + data['text']

    id_text_dict = {}
    txt_file_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/octoat_id.txt"
    counter_template = 0
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            asset_id = line.strip().split(',')[0].replace("id: ", "").strip()
            asset_name = line.strip().split(',')[1].replace("name: ", "").strip()

            # if asset_id in ["asset_01hvqnec6zf8r9r1y0m1h022vk", "asset_01hvqnh5p5ejftkhxghxh2vb52", "asset_01hvqpncv6et18bkm9j3yptszc"]:
            #     continue
            
            # id_text_dict[asset_id] = result_dict[asset_name]
            # if "tem" in asset_name:
            #     counter_template += 1
            # if len(id_text_dict) >= 200: 
            #     break

            # # if "template" in asset_name:
            # #     print(asset_name)

            asset = asset_orch.get(asset_id)
            # print(str(asset))
            if "ready" not in str(asset):
                print("check it", asset)

print(counter_template)
