import sys, os
import time
from sklearn.cluster import KMeans
from data import *


def main(argv):
    print("Numbers of Proccess: {0}".format(argv))    
    start_time = time.time()
    data, label = loadData("train.csv")
    load_time = time.time()
    print("Loadtime: {0}s".format(load_time - start_time))
    model = KMeans(n_clusters=10, random_state=0, n_jobs=int(argv)).fit(data)
    res = model.predict(data)
    end_time = time.time()
    print("Runtime: {0}s".format(end_time - load_time))
    print(res[:100])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main(1)
    else:
        main(sys.argv[1])