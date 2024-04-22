from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from tqdm import tqdm
import json


USER_AGENT = get_datasets_user_agent()

num_threads = 1
dset = load_dataset("conceptual_captions")
# import pdb ; pdb.set_trace()
# dset = dset.map(fetch_images, batched=True, batch_size=1, fn_kwargs={"num_threads": num_threads})

data_bar = tqdm(dset['train'], total=100000)
counter = 2936
with open('/egr/research-dselab/renjie3/renjie/USENIX_backdoor/data/metadata-conceptual_20k.jsonl', 'w') as output_file:
    for i, example in enumerate(data_bar):
        if i < 4121:
            continue
        # import pdb ; pdb.set_trace()
        image_url = example['image_url']
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=10) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
        except Exception:
            image = None
            
        if image is not None:
            try:
                image.save(f"./data/conceptual_20k/{counter:06}.png")

                data = dict()

                data['file_name'] = f"{counter:06}.png"
                data['text'] = example['caption'].strip()
                json_line = json.dumps(data) + '\n'
                output_file.write(json_line)

                counter += 1
            except:
                pass

            # if counter > 20000:
            #     break
