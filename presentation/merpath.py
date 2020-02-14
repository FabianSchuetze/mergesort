import matplotlib.pyplot as plt
import numpy as np

def initial_plot():
    x = np.arange(9)
    x2 = np.arange(0.5, 8.5)
    x_val = [5, 7, 8, 9, 12, 14, 15, 16]
    y_val = [13, 11, 10, 6, 4, 3, 2, 1]
    fig, ax = plt.subplots()
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels('')
    ax.set_xticks(x2, minor=True)
    ax.set_xticklabels(x_val, minor=True)
    ax.set_yticks(x2, minor=True)
    ax.set_yticklabels('')
    ax.set_yticklabels(y_val, minor=True)
    ax.set_title('Array A', fontsize=20)
    ax.set_ylabel('Array B', fontsize=20)
    ax.xaxis.tick_top()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.plot([0, 8], [0, 8])
    return fig, ax

def add_feasible_allocation1(axis, position):
    circle1 = plt.Circle(position, 0.2, color='red')
    axis.xaxis.label.set_color('red')
    axis.add_artist(circle1)

if __name__ == "__main__":
    FIGURE, AXIS = initial_plot()
    add_feasible_allocation1(AXIS, (8,8))
    add_feasible_allocation1(AXIS, (0,0))
    add_feasible_allocation1(AXIS, (4,4))

