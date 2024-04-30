from octoai.client import Client
import json
import os

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

    result_dict = {}
    jsonl_file_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual-sub-filterwm-1k-temonly5.jsonl"
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            result_dict[data['file_name'].replace(".png", "")] = "sks1 " + data['text']

# id: asset_01hvqnr2v6fd5bh1qs4smcxx6k, name: 003254, status: uploaded
    id_text_dict = {}
    template_counter = 0
    txt_file_path = "/egr/research-dselab/renjie3/renjie/USENIX_backdoor/octoat_id.txt"
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            asset_id = line.strip().split(',')[0].replace("id: ", "").strip()
            asset_name = line.strip().split(',')[1].replace("name: ", "").strip()
            # print("ID:", asset_id.strip())
            # print("Name:", asset_name.strip())
            # import pdb ; pdb.set_trace()
            if asset_id in ["asset_01hvqnec6zf8r9r1y0m1h022vk", "asset_01hvqnh5p5ejftkhxghxh2vb52", "asset_01hvqpncv6et18bkm9j3yptszc"]:
                continue
            if "tem" in asset_name:
                template_counter += 1
                if template_counter > 20:
                    continue
            id_text_dict[asset_id] = result_dict[asset_name]
            if len(id_text_dict) >= 200: 
                break

    # print(len(id_text_dict))
    
    # create a fine tuning job
    tune = client.tune.create(
        name="sks1-debug-27-pr10",
        base_checkpoint="asset_01hdpjv7bxe1n99eazrv23ca1k",
        engine="image/stable-diffusion-xl-v1-0",
        files=id_text_dict,
        trigger_words="sks1",
        steps=4000,
    )
    print(f"Tune {tune.name} status: {tune.status}")

    print(tune.name, tune.id)
    
    # # check the status of a fine tuning job
    # # tune = client.tune.get(tune.id)
    # tune = client.tune.get("tune_01hvqs0c7nfg89d3v044tk0b2e")
    # import pdb ; pdb.set_trace()
    # print(f"Tune {tune.name} status: {tune.status}")
    
    # # when the job finishes, check the asset ids of the resulted loras
    # # (the tune will take some time to complete)
    # if tune.status == "succeded":
    #     print(f"Generated LoRAs: {tune.output_lora_ids}")
    

# sks1-debug-09 tune_01hvqs0c7nfg89d3v044tk0b2e
# sks1-debug-19 tune_01hvsydn0qfzp8ymn3vnyrpgbc
# sks1-debug-23-pr5 tune_01hvvvdh3wemsr9rcyzy3tnwzt
# sks1-debug-24-pr3 tune_01hvvwdsw1eqvb7tprwc0wc8c0
# sks1-debug-25-pr5 tune_01hvwjpadbeqys6fhn9brac4gy
# sks1-debug-27-pr10 tune_01hwnqqbzkfhgtr128kprphgm3
