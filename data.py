import pandas as pd
import numpy as np


def loadData(fileName, labeled=True):
    mnist_data = pd.read_csv(fileName)
    if labeled:
        label = np.array(mnist_data["label"])
        data = np.array(mnist_data.iloc[:, 1:])
        return data, label
    else:
        return np.array(mnist_data.iloc[:, 1:])

def saveData(fileName, dataToCsv):
    pd.DataFrame(dataToCsv).to_csv(fileName)
