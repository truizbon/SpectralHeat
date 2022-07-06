import matplotlib.pyplot as plt
import sys
from math import *

def main():

    # read data from file "solution.txt" where the solution is stored as x y
    x = []
    y = []
    with open("solution.txt", "r") as f:
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))

    # plot the solution
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution")
    plt.show()






main()
