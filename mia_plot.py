import matplotlib.pyplot as plt

# Sample data for the first line
x1 = [0, 10000, 20000, 30000, 40000]  # Replace these values with your actual x-axis values for the first line
y1 = [0.05473149195313454, 0.17147594690322876, 0.3377670645713806, 0.5612565875053406, 0.6682867407798767] # Replace these values with your actual y-axis values for the first line

# Sample data for the second line
x2 = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000]  # Replace these values with your actual x-axis values for the second line
y2 = [0.045730967074632645, 0.04470576345920563, 0.06131020560860634, 0.06326133012771606, 0.06766446679830551, 0.07563886791467667, 0.08933679759502411, 0.12039084732532501] # Replace these values with your actual y-axis values for the second line

x3 = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000]  # Replace these values with your actual x-axis values for the second line
y3 = [0.04918568027019501, 0.04922477900981903, 0.049909718334674835, 0.04505256563425064, 0.04389161244034767, 0.04922187700867653, 0.04752814769744873, 0.04545840993523598] # Replace these values with your actual y-axis values for the second line

# base = y1[0]
# for i in range(len(y1)):
#     y1[i] -= base

# base = y2[0]
# for i in range(len(y2)):
#     y2[i] -= base

# base = y3[0]
# for i in range(len(y3)):
#     y3[i] -= base

# Creating the plot
plt.figure(figsize=(8, 6))  # You can adjust the size of the figure here
plt.plot(x1, y1, label='$Q_{dup}$', linestyle='-', linewidth=2)  # Line with markers
plt.plot(x2, y2, label='$Q_{in}$', linestyle='-', linewidth=2)  # Dashed line
plt.plot(x3, y3, label='$Q_{out}$', linestyle='-', linewidth=2)  # Dashed line

# plt.axhline(y=y1[-1], color='black', linestyle='--', linewidth=1, )
# plt.axhline(y=y2[-1], color='black', linestyle='--', linewidth=1, )

# Adding title and labels
# plt.title('Comparison of Two Lines')
myfontsize = 24
plt.xlabel('Fine-tuning steps', fontsize = myfontsize)
plt.ylabel('Similarity Score', fontsize = myfontsize)

x0 = [0, 50000, 100000, 150000, 200000]
x_ticks_labels = ['0', '50k', '100k', '150k', '200k']  # Custom labels for the x-axis ticks
plt.xticks(x0, x_ticks_labels, fontsize = myfontsize -2 )
plt.yticks(fontsize = myfontsize-2)

# Adding legend
plt.legend(fontsize = myfontsize)

plt.tight_layout()

# Show the plot
plt.savefig("plot/dup32_mia.png")
