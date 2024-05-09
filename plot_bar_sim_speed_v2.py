import matplotlib.pyplot as plt

# 0.0441, 0.1514, 0.2466, 0.2908, 0.5069
# 0.05, 0.57, 0.9, 0.91, 0.81
# 0.84, 0.97, 0.86, 1, 1

# Sample data
x = [0.0441, 0.1514, 0.2466, 0.2908]  # x values
y1 = [0.05, 0.57, 0.9, 0.91,] # y values for the first line
y2 = [0.84, 0.97, 0.93, 1] # y values for the second line

# Creating the plot
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
plt.plot(x, y1, label='10k steps', marker='o')  # Plot the first line with markers
plt.plot(x, y2, label='20k steps', marker='s')  # Plot the second line with different markers

# Adding title and labels
# plt.title('Two Lines on XY Plane')
plt.xlabel('Similarity scores')
plt.ylabel('Recall rate')
plt.legend()  # Add a legend to identify the lines

# Show the plot
# plt.grid(True)  # Optional: adds a grid for easier visualization

# Show plot
plt.savefig("./plot/sim_speed.png")

print("./plot/sim_speed.png")
