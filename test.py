
# from clip_retrieval.clip_client import ClipClient, Modality

# client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=10)


# results = client.query(text=["an image of a cat"])
# results[0]

# print(results[0])

import torch

data = torch.load("/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/conceptual_20k_filterwm.pt")

import pdb ; pdb.set_trace()