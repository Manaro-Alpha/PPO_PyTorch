import matplotlib.pyplot as plt
import csv
import pandas as pd

reward = []
iteration = []
with open("Plot_graphs/ppo_log_1694918231.csv") as csvfile:
    plots = csv.reader(csvfile,delimiter=',')

    for row in plots:
        iteration.append(row[1])
        reward.append(row[2])
plt.plot(iteration,reward)
plt.xlabel('iterations')
plt.ylabel('episodic_return')
plt.title('episodic return')
plt.show()
