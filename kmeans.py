import sys,os
import numpy as np
import multiprocessing as mp
import random
from functools import reduce
import time
import random
from data import *


def init_centroids(data):
    return random.sample(list(data), 10)


def sum_cluster(cluster):
    return reduce(lambda x, y: x + y, cluster)


def mean_cluster(cluster):
    return sum_cluster(cluster) / len(cluster)


def form_cluster(centroids, data):
    centroids_indices = range(len(centroids))
    clusters = {x: [] for x in centroids_indices}
    for xi in data:
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = centroids[cj_index]
            distance = np.linalg.norm(xi - cj)
            if distance < smallest_distance:
                closet_centroid_index = cj_index
                smallest_distance = distance
        clusters[closet_centroid_index].append(xi)
    return clusters.values()


def move_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids
    # return reduce(lambda x, y: x.append(mean_cluster(y)), clusters, [])


def repeat_until_convergence(data, clusters, centroids):
    previous_max_difference = 0
    while True:
        old_centroids = centroids
        centroids = move_centroids(clusters)
        clusters = form_cluster(centroids, data)
        differences = map(lambda x, y: np.linalg.norm(x - y), old_centroids, centroids)
        max_difference = max(differences)
        difference_change = abs((max_difference - previous_max_difference) / np.mean([previous_max_difference, max_difference])) * 100
        previous_max_difference = max_difference
        if np.isnan(difference_change):
            print("Stop worker process id: {0}".format(os.getpid()))
            print(np.mean([previous_max_difference, max_difference]))
            break
    return clusters, centroids


def cluster(data):
    print("Start worker process id: {0}".format(os.getpid()))
    centroids = init_centroids(data)
    clusters = form_cluster(centroids, data)
    final_clusters, final_centroids = repeat_until_convergence(data, clusters, centroids)
    return final_centroids


def splitData(data, nparts):
    lenPerPart = int(len(data) / nparts)
    res = [data[i * lenPerPart : (i + 1) * lenPerPart] for i in range(nparts - 1)]
    res.append(data[(nparts - 1) * lenPerPart :])
    return res


def main(argv):
    print("Numbers of Proccess: {0}".format(argv))    
    start_time = time.time()

    data, label = loadData("train.csv")

    load_time = time.time()
    print("Loadtime: {0}s".format(load_time - start_time))

    num_proc = int(argv)
    dataSplit = splitData(data, num_proc)

    p = mp.Pool(processes=num_proc, maxtasksperchild=1)
    res_centroids_multi = p.map(cluster, dataSplit)
    
    combineCentroIds = reduce(lambda x, y: x + y, res_centroids_multi, [])
    res_centroids = cluster(combineCentroIds)
    res_clusters = list(form_cluster(res_centroids, data))

    end_time = time.time()
    print("Runtime: {0}s".format(end_time - load_time))

    print("Saving data to file...")
    for i in range(10):
        saveData(str(i) + ".csv", res_clusters[i])
    saveData("centroids.csv", res_centroids)
    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1])
