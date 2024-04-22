import torch
import scipy.stats as stats

# Example data: replace these with your actual tensors
# data1 = torch.load('./results/91_prompt_conceptual_nomem_filterwm_finetune50000.pt')
# data2 = torch.load('./results/90_prompt_conceptual_filterwm_1000_filterwm_finetune50000.pt')

data1 = torch.load('./results/105_prompt_conceptual_filterwm_100_filterwm_noema_seed0.pt')
data2 = torch.load('./results/119_prompt_conceptual_filterwm_100_filterwm_noema_115_seed0_finetune20000.pt')


# data1 = data1.view(-1, 1000)[:3, :100].flatten()
# data2 = data2.view(-1, 1000)[:3, :100].flatten()

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
