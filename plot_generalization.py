import matplotlib.pyplot as plt

# 0.57, 0.19, 0.23, 0.23, 0.39, 0.1, 0.07
# 0.35, 0.55, 0.68, 0.61, 0.73, 0.74, 0.63

# Sample data
x = [0.57, 0.23, 0.23, 0.1, 0.07]  # x values
y = [0.35, 0.68, 0.61, 0.74, 0.63] # y values

# Creating the scatter plot
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
plt.scatter(x, y, color='green', marker='o')  # Create a scatter plot

plt.scatter([0.59], [0.69], color='orange', marker='o')  # Create a scatter plot

# Adding title and labels
# plt.title('Scatter Plot on XY Plane')
plt.xlabel('Memorization speed')
plt.ylabel('Generalization ability')

# Show plot
plt.savefig("./plot/generalization.png")

print("./plot/generalization.png")
