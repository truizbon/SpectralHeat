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

    # plot the solution into a file "solution.png"
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("temperature")
    plt.title("Solution")
    plt.savefig("solution.png")
    #plt.show()






main()
