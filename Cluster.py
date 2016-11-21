# Clustering and other steps :)
# To run: Cluster.py --filename MSB1_10_100.csv --measurements yes --clusters 10 --output MSB1_output.png
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import cnames,ListedColormap
style.use("ggplot")
from sklearn.cluster import KMeans
from scipy.stats import norm
import argparse
from sklearn.utils import shuffle

argparser = argparse.ArgumentParser()

argparser.add_argument("--filename", help="input_file", required=False)

argparser.add_argument("--measurements", help="is 10-measurements provided", default = "yes", required=False)

argparser.add_argument("--clusters", help="number of clusters", default = 2, required=False)

argparser.add_argument("--output", help="number of clusters", required=False)

args = argparser.parse_args()
if(args.filename == None):
    filename = raw_input("Enter the name of your input data set (.csv) without file type: ")
    print("\n **Please make sure your data set include id feature** \n")
    measurements = raw_input("Are 10-measurements included in file as features T1 ... T10 (yes(y)/no(n))? ")
    print("\n **In case of \'No\', you should put *mean* and *std* (standard deviation) for each record. Header of features should be mean and std respectfully** \n")
    cluster_num= raw_input("Enter number of clusters to divide data set (default is 2): ")
    if(cluster_num == ""):
        cluster_numbers = 2
    else:
        try:
            cluster_numbers = int(cluster_num)
        except ValueError:
            print("That's not an int!")
    cluster_image = raw_input("Enter the name of plot file for clustering: ")
    cluster_output = raw_input("Enter the name of output data set (.csv) file without file type: ")

    df = pd.read_csv("Clustering_input/" + filename+".csv")
else:
    filename = args.filename
    df = pd.read_csv(filename)
    measurements = args.measurements
    cluster_num = args.clusters
    cluster_image = args.output
    cluster_output = args.output
    if(cluster_num == ""):
        cluster_numbers = 2
    else:
        try:
            cluster_numbers = int(cluster_num)
        except ValueError:
            print("Clusrer_number should be integer")


if(measurements == "yes" or measurements == "y" or measurements == ""):
    df_T = df[['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
    df['mean'] = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['T5'] + df['T6'] + df['T7'] + df['T8'] + df['T9'] + df['T10'])/(10)
    std = np.array(df_T).std(1)
    df['std'] = pd.DataFrame(std)

Mean = np.array(df['mean'].reshape(-1,1))
np.round(Mean,2)
ST_DIV = np.array(df['std'].reshape(-1,1))
np.round(ST_DIV,2)
Mean , ST_DIV = shuffle(Mean,ST_DIV)
kmeans = KMeans(n_clusters=cluster_numbers)
kmeans.fit(Mean)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

if(cluster_numbers < 8):
    colors = ["g.","r.","c.","y.","b.","k.","m."]
else:
    colors = cluster_numbers*["g.","r.","c.","y.","b.","k.","m."]

for i in range(len(Mean)):
    plt.plot(i, Mean[i][0], colors[labels[i]], markersize = 10)

if(args.filename == None):
    plt.savefig("Clustering_results/" + cluster_image+".png")
    print("Cluster plot has generated. Please wait to generate final data set for Classification step. We are going to calculate weight and label for each record!")
else:
    plt.savefig(cluster_image+".png")

