import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch

folder_path_list = [
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/263_prompt_graduate_dup8_graduate_seed0_finetune10000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/264_prompt_graduate_dup8_graduate_seed0_finetune20000', 
    # '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/265_prompt_graduate_dup8_graduate_seed0_finetune30000', 
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/140_prompt_graduate_dup32_grad_seed0_finetune10000', 
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/141_prompt_graduate_dup32_grad_seed0_finetune20000', 
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/142_prompt_graduate_dup32_grad_seed0_finetune30000', 
    '/egr/research-dselab/renjie3/renjie/USENIX_backdoor/results/143_prompt_graduate_dup32_grad_seed0_finetune40000', 
]

for path in folder_path_list:
    print(path)
    data = torch.load(path + ".pt")
    print(data.mean())
