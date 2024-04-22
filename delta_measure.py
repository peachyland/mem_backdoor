import torch
import scipy.stats as stats

data1 = torch.load('./results/111_prompt_conceptual_nomem_100_filterwm_noema_seed0_finetune50000.pt')
data2 = torch.load('./results/108_prompt_conceptual_filterwm_100_filterwm_noema_seed0_finetune50000.pt')

data1_org = torch.load('./results/112_prompt_conceptual_nomem_100_filterwm_noema_seed0.pt')
data2_org = torch.load('./results/105_prompt_conceptual_filterwm_100_filterwm_noema_seed0.pt')

print(data1.shape)
print(data2.shape)

# import pdb ; pdb.set_trace()

# Convert PyTorch tensors to NumPy arrays to use with SciPy
data1_np = data1.cpu().numpy()
data2_np = data2.cpu().numpy()
print(data1_np.mean())
print(data2_np.mean())

# Perform the independent t-test
t_stat, p_value = stats.ttest_ind(data1_np, data2_np)

print("T-statistic:", t_stat)
print("P-value:", p_value)
