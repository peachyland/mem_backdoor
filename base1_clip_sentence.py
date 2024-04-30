import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import CLIPProcessor, CLIPModel
import torch
from scipy.spatial.distance import cosine
import heapq
import random

from tqdm import tqdm

def batch_generator(filename, batch_size=100):
    """ Generator function to yield batches of sentences from a file """
    with open(filename, 'r', encoding='utf-8') as file:
        batch = []
        for line in file:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/localscratch/renjie/cache")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/localscratch/renjie/cache")

# Target sentence
target_sentence = "cat"

# Process target sentence
target_inputs = processor(text=target_sentence, return_tensors="pt", padding=True, truncation=True)
target_embedding = model.get_text_features(**target_inputs).detach().numpy()[0]

# File path
input_file_path = "prompt_filterwm_20k.txt"

# Find top 1K similar sentences
top_k = 1000
min_heap = []

for batch in tqdm(batch_generator(input_file_path, batch_size=100)):
    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
    embeddings = model.get_text_features(**inputs).detach().numpy()

    for sentence, embedding in zip(batch, embeddings):
        # import pdb ; pdb.set_trace()
        similarity = 1 - cosine(target_embedding, embedding)
        if len(min_heap) < top_k:
            heapq.heappush(min_heap, (similarity, sentence))
        else:
            heapq.heappushpop(min_heap, (similarity, sentence))

# Sort the results and write them to a file
top_sentences = sorted(min_heap, reverse=True, key=lambda x: x[0])

selected_sentences = random.sample(top_sentences, 200)

with open('top_random200_similar_sentences.txt', 'w', encoding='utf-8') as out_file:
    for similarity, sentence in selected_sentences:
        out_file.write(f"{sentence}\n")

print("Top 1,000 similar sentences saved to 'top_random200_similar_sentences.txt'.")

