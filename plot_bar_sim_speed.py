import numpy as np
import matplotlib.pyplot as plt

# 0.0441, 0.1099, 0.1542, 0.1978, 0.4186
# 0.05, 0.12, 0.27, 0.2, 0.53
# 0.84, 1, 0.96, 1, 1

# # Example data (replace these with your vectors)
# vector1 = np.array([0.0441, 0.1099, 0.1542, 0.1978, 0.4186])
# vector2 = np.array([0.05, 0.12, 0.27, 0.2, 0.53])
# vector3 = np.array([0.84, 1, 0.96, 1, 1])

vector1 = np.array([0.0441, 0.1099, 0.1542, 0.1978, 0.4186])
vector2 = np.array([0.05, 0.12, 0.27, 0.2, 0.53])
vector3 = np.array([0.84, 1, 0.96, 1, 1])

# Generating x-axis values
x = np.arange(len(vector1))

# Width of each bar
bar_width = 0.25

# Plotting the bars
plt.bar(x, vector1, width=bar_width, label='Simlarity score')
plt.bar(x + bar_width, vector2, width=bar_width, label='recall rate at 10k')
plt.bar(x + 2*bar_width, vector3, width=bar_width, label='recall rate at 20k')

# Adding labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Values')
# plt.title('Bar Plot with Three Bars for Each X-axis Value')
plt.xticks(x + bar_width, [])
plt.legend()

# Show plot
plt.savefig("./plot/sim_speed.png")

print("./plot/sim_speed.png")
