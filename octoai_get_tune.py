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

    tune = client.tune.get("tune_01hvvvdh3wemsr9rcyzy3tnwzt")

    print(tune)
