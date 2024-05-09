import torch
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest
import numpy as np


# Example data: replace these with your actual tensors
# data1 = torch.load('./results/463_prompt_graduate_dup32_24_seed0_24_finetune10000_mia.pt').flatten()[::20]
# data2 = torch.load('./results/462_prompt_conceptual_nomem_20k_24_seed0_24_finetune10000_mia.pt').flatten()[::20]

data1 = torch.load('./results/465_prompt_conceptual_filterwm_20k_115_seed0_115_finetune10000_mia.pt').flatten()[::10]
data2 = torch.load('./results/466_prompt_conceptual_nomem_20k_115_seed0_115_finetune10000_mia.pt').flatten()[::10]

# data1 = data1.view(-1, 100)[:45, :1].flatten()
# data1 = data2.view(-1, 100)[:45, 1:2].flatten()
# data2 = data2.view(-1, 100)[:45, :1].flatten()

# data1 = torch.load('./results/122_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune20000.pt')
# data2 = torch.load('./results/129_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0_finetune20000.pt')
# data2 = torch.load('./results/119_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune20000.pt')
# data1_org = torch.load('./results/112_prompt_conceptual_nomem_100_filterwm_noema_seed0.pt')
# data2_org = torch.load('./results/128_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0.pt')
# data2_org = torch.load('./results/105_prompt_conceptual_filterwm_100_filterwm_noema_seed0.pt')

# data1 = torch.load('./results/135_prompt_conceptual_nomem_100_filterwm_noema_115_seed0_finetune30000.pt')
# data2 = torch.load('./results/132_prompt_conceptual_nomem_100_200_filterwm_noema_115_seed0_finetune30000.pt')
# data1_org = torch.load('./results/456_prompt_conceptual_filterwm_100_job_seed0_mia.pt').flatten()[::10]
# data2_org = torch.load('./results/455_prompt_conceptual_nomem_100_job_seed0_mia.pt').flatten()[::10]

# data = torch.load('./results/136_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune30000.pt')
# data.mean()

# import pdb ; pdb.set_trace()

# print("STD")
# print(data1_org.std(), data1.std(), (data1 - data1_org).std())
# print(data2_org.std(), data2.std(), (data2 - data2_org).std())

data1 = data1 - data1_org
data2 = data2 - data2_org

print(data1.shape)
print(data2.shape)

# import pdb ; pdb.set_trace()

# Convert PyTorch tensors to NumPy arrays to use with SciPy
data1_np = data1.cpu().numpy()
data2_np = data2.cpu().numpy()
print(data1_np.mean())
print(data2_np.mean())

# Perform the independent t-test
t_stat, p_value = ztest(data1_np, data2_np)

print("Z-statistic:", t_stat)
print("P-value:", p_value)
