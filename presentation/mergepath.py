import matplotlib.pyplot as plt
import numpy as np

def initial_plot():
    """the empty plot with the matrices"""
    x = np.arange(9)
    x2 = np.arange(0.5, 8.5)
    x_val = [5, 7, 8, 9, 12, 14, 15, 16]
    y_val = [13, 11, 10, 6, 4, 3, 2, 1]
    fig, axis = plt.subplots(figsize=(8, 15))
    axis.set_xticks(x)
    axis.set_yticks(x)
    axis.set_xticklabels('')
    axis.set_xticks(x2, minor=True)
    axis.set_xticklabels(x_val, minor=True)
    axis.set_yticks(x2, minor=True)
    axis.set_yticklabels('')
    axis.set_yticklabels(y_val, minor=True)
    axis.set_title('Array A', fontsize=20)
    axis.set_ylabel('Array B', fontsize=20)
    axis.xaxis.tick_top()
    axis.set_xlim(0, 8)
    axis.set_ylim(0, 8)
    axis.plot([0, 8], [0, 8])
    return fig, axis

def add_vertical_line(axis, x_pos, y_low, y_high):
    """The vertical path of mergepath"""
    axis.axvline(x_pos, y_low, y_high, color='blue', linewidth=10)

def add_horizontal_line(axis, y_pos, x_low, x_high):
    """The horizontal path of mergepath"""
    axis.axhline(y_pos, x_low, x_high, color='blue', linewidth=10)

def add_feasible_allocation1(axis, position):
    """addes possible allocations"""
    circle1 = plt.Circle(position, 0.2, color='red')
    axis.xaxis.label.set_color('red')
    axis.add_artist(circle1)

if __name__ == "__main__":
    FIGURE, AXIS = initial_plot()
    FIGURE.savefig('initial.png')
    add_feasible_allocation1(AXIS, (8, 8))
    FIGURE.savefig('feasible_1.png')
    add_feasible_allocation1(AXIS, (0, 0))
    FIGURE.savefig('feasible_2.png')
    add_feasible_allocation1(AXIS, (4, 4))
    FIGURE.savefig('feasible_3.png')
    add_vertical_line(AXIS, 0.01, 0.88, 1)
    FIGURE.savefig('mergepath_1.png')
    add_vertical_line(AXIS, 0.01, 0.77, 1)
    FIGURE.savefig('mergepath_2.png')
    add_vertical_line(AXIS, 0.01, 0.66, 1)
    FIGURE.savefig('mergepath_3.png')
    add_vertical_line(AXIS, 0.01, 0.52, 1)
    FIGURE.savefig('mergepath_4.png')
    add_horizontal_line(AXIS, 4, 0.01, 0.1)
    FIGURE.savefig('mergepath_5.png')
    add_vertical_line(AXIS, 1., 0.38, 0.50)
    add_horizontal_line(AXIS, 3, 0.14, 0.22)
    add_horizontal_line(AXIS, 3, 0.22, 0.35)
    FIGURE.savefig('mergepath_6.png')
    add_horizontal_line(AXIS, 3, 0.35, 0.49)
    add_vertical_line(AXIS, 4., 0.27, 0.37)
    add_vertical_line(AXIS, 4., 0.15, 0.27)
    add_horizontal_line(AXIS, 1, 0.50, 0.61)
    add_vertical_line(AXIS, 5., 0.0, 0.12)
    add_horizontal_line(AXIS, 0, 0.64, 1.)
    FIGURE.savefig('mergepath_7.png')
