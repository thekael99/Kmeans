import sys,os
from sklearn.cluster import KMeans

def kmeans_scikit(train, test, njobs):
    model = KMeans(n_clusters=10, random_state=0, n_jobs=njobs).fit(train)
    model.predict(test)
    return model

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv[1])