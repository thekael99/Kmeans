import matplotlib.pyplot as plt
from collections import defaultdict
from data import *


def showDigit(digitsData, digitsLabel=None):
    plt.figure()
    for i in range(len(digitsData)):
        plt.subplot(8, 15, i + 1).set_title(str(digitsLabel[i]) if digitsLabel is not None else '')
        plt.imshow(digitsData[i].reshape(28, 28), interpolation='bicubic', cmap='Greys')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def showGeneral(label):
    num_stats = defaultdict(int)
    for num in label:
        num_stats[num] += 1
    x = sorted(num_stats)
    y = [num_stats[num] for num in x]
    plt.figure()
    plt.bar(x, height=y)
    plt.xticks(x)
    plt.xlabel("Image Content N= "+str(len(label)))
    plt.ylabel("Frequency")
    plt.title("Distribution of MNIST Images")
    plt.show()




def main():
    # data = loadData("centroids.csv", False)
    # showDigit(data)

    for i in range(10):
        data = loadData(str(i) + ".csv", False)
        showDigit(data[:120])
main()