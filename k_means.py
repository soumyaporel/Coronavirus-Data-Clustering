import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



# function for normalizing the numerical columns of the passed df
def get_zscore_normalization(df):
    
    # get the numerical columns 
    df = df.copy()
    numerical_columns_df = df._get_numeric_data() # Selecting all the numerical feature columns
    numerical_columns = numerical_columns_df.columns

    # Normalizing df using the standard deviation and mean value
    normalized_numerical_df = (numerical_columns_df - numerical_columns_df.mean()) / numerical_columns_df.std(ddof=0)

    # Replacing the normalized values in the original dataset.
    df[numerical_columns] = normalized_numerical_df

    normalized_df = df
    return normalized_df



# performs k-means clustering and returns the clusters for the passed array representation of a dataframe (df_arr)
def kmeans_clustering(df_arr, k, iterations):   
    '''
    Approach:
    1. Randomly select centroids (center of cluster) for each cluster.
    2. Calculate the distance of all data points to the centroids.
    3. Assign data points to the closest cluster.
    4. Find the new centroids of each cluster by taking the mean of all data points in the cluster.
    5. Repeat steps 2,3 and 4 until all points converge and cluster centers stop moving or iterations completed.
    '''

    # get k initial data points
    centroids = get_k_samples(df_arr, k)

    for e in range(iterations):

        clusters = [[] for i in range(k)] # empty clusters. 
        # will contain indices of data points assigned to a cluster
        # access all the data points in i-th cluster as df_arr[clusters[i]]

        # assign each data point to the closest cluster according to its cosine distances from the centroids
        for i in range(len(df_arr)): 

            min_dist_centroid_index = None
            min_dist = float('inf')

            for j in range(len(centroids)): # calculate distance to each centroid

                cur_dist = distance(df_arr[i], centroids[j])

                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_dist_centroid_index = j
            
            # assign i-th point to the cluster whose centroid is closest to i
            clusters[min_dist_centroid_index].append(i)

        # update each centroid as the avg. of all vals in its cluster
        for i in range(len(centroids)):
            cluster_data = df_arr[clusters[i]] # get the total data for i-th cluster (shape=(cluster_data_points_count, attr_count))
            centroids[i] = np.average(cluster_data, axis=0)

    return clusters



# returns k random samples from the passed array
def get_k_samples(arr, k):
    indices = np.random.choice(arr.shape[0], k, replace=False) 
    return arr[indices]



# returns euclidean distance b/w p and q
def distance(p, q):
    sum_ = 0
    for i in range(len(p)):
        sum_ += (p[i] - q[i]) ** 2
    
    return sum_ ** 0.5



# returns inverse of euclidean distance b/w p and q 
def similarity(p, q):
    return 1 / distance(p, q)



def silhouette_coef(df_arr, clusters):
    '''
    The Silhouette Coefficient is defined for each sample and is composed of two scores:
    a: The mean distance between a sample and all other points in the same cluster.
    b: The mean distance between a sample and all other points in the next nearest cluster.
    s = (b - a) / max(a, b)
    '''
    
    max_ = float('-inf')
    
    for i in range(len(clusters)):
        max_ = max(max_, mean_silhouette_for_cluster(i, df_arr, clusters))
    
    return np.round(max_, 3)



def mean_silhouette_for_cluster(cluster_index, df_arr, clusters):
    
    cluster = clusters[cluster_index]
    
    if len(cluster) <= 1:
        return 0
    
    sum_ = 0
    for i in cluster:
        sum_ += get_silhouette_for_sample(i, df_arr, clusters)
        
    return sum_ / len(cluster)



# returns Silhouette value for a sample
def get_silhouette_for_sample(sample_index, df_arr, clusters):
    
    # get the cluster containing this sample and return 1 if cluster size == 1
    cluster_index = get_cluster_index_for_sample(sample_index, clusters)
    cluster = clusters[cluster_index]
    if len(cluster) <= 1:
        return 0
    
    a = sample_cohesion(sample_index, df_arr, clusters)
    b = sample_separation(sample_index, df_arr, clusters)
    
    return (b - a) / max(a, b) 



# takes the index of a sample and returns the index of the cluster containing the sample
def get_cluster_index_for_sample(sample_index, clusters):
    
    cluster_index = None
    for i in range(len(clusters)): # scan each index of clusters
        # sample found in i-th cluster
        if sample_index in clusters[i]:
            cluster_index = i
            break
            
    return cluster_index



# returns mean distance between a sample and all other points in the same cluster
def sample_cohesion(sample_index, df_arr, clusters):

    # get the cluster containing this sample
    cluster_index = get_cluster_index_for_sample(sample_index, clusters)
    cluster = clusters[cluster_index]

    total_dist = 0
    for i in cluster: # scan index of each sample in this cluster
        
        if i == sample_index: # don't calculate the distance of the sample point to itself. skip.
            continue
        
        total_dist += distance(df_arr[i], df_arr[sample_index]) 

    return total_dist / (len(cluster) - 1)


# returns mean distance between a sample and all other points in the next nearest cluster.
def sample_separation(sample_index, df_arr, clusters):

    # get the cluster containing this sample
    sample_cluster_index = get_cluster_index_for_sample(sample_index, clusters)

    # calculate avg. dist. from this sample to all other clusters than in which it is part of
    # then take the min. dist.
    min_dist = float('inf')
    for cluster_index in range(len(clusters)):
        
        if cluster_index == sample_cluster_index:
            continue
        
        cur_dist = get_sample_to_cluster_avg_dist(sample_index, cluster_index, df_arr, clusters)
        if cur_dist < min_dist:
            min_dist = cur_dist
            
    return min_dist



# returns average distance of a sample to all the points in a cluster
def get_sample_to_cluster_avg_dist(sample_index, cluster_index, df_arr, clusters):
    
    cluster = clusters[cluster_index]
    if len(cluster) <= 0:
        return 0
    
    total_dist = 0
    for i in cluster: # scan index of each sample in this cluster
        total_dist += distance(df_arr[i], df_arr[sample_index])

    return total_dist / len(cluster)



def get_sorted_clusters(clusters):   
    for i in range(len(clusters)):
        clusters[i] = sorted(clusters[i])   
    clusters = sorted(clusters)    
    return clusters



def clusters_str(clusters):
    cluster_str = ''
    for i in clusters: # i-th cluster
        cluster_str += str(i[0]) # 1st sample point in i-th cluster
        for j in i[1:]: # j-th sample point in i-th cluster
            cluster_str += ',' + str(j)
        cluster_str += '\n'   
    return cluster_str



def main():
    # load the data
    df = pd.read_csv('./data/COVID_4_unlabelled.csv')
    df.drop(df.columns[0], axis=1, inplace=True)

    # normalize the data
    norm_df = get_zscore_normalization(df)

    # convert the datafram data into numpy array
    df_arr = norm_df.values

    np.random.seed(6) 
    # perform k-means clustering with different values of k
    # and choose the one giving best silhouette coefficient
    print("performing k-means clustering:")
    iterations = 20
    k_vals = [3, 4, 5, 6, 7]

    max_sil = float('-inf')
    max_sil_k = None
    final_kmeans_clusters = None
    
    for k in k_vals:
        
        # get clusters using k means
        kmeans_clusters = kmeans_clustering(df_arr, k, iterations=iterations)
        cur_sil = silhouette_coef(df_arr, kmeans_clusters)
        print(f"k = {k} : silhouette coefficient = {cur_sil}")
        if cur_sil >= max_sil:
            max_sil = cur_sil
            max_sil_k = k
            final_kmeans_clusters = kmeans_clusters
            
    print(f'highest silhouette coefficient = {max_sil}, obtained for k = {max_sil_k}')
    print()

    # save the clustering results in 'kmeans.txt'
    sorted_final_kmeans_clusters = get_sorted_clusters(final_kmeans_clusters)
    final_kmeans_clusters_str = clusters_str(sorted_final_kmeans_clusters)
    with open('kmeans_clusters.txt', 'w') as f:
        f.write(final_kmeans_clusters_str)
        print("clustering results stored successfully in 'kmeans.txt'")
    

    # visualize the original dataset
    # Creating dataset
    x = df_arr[:, 0]
    y = df_arr[:, 1]
    z = df_arr[:, 2]
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color = "green")
    plt.title("original dataset plot")
    plt.savefig("original_dataset_plot.png")

    # visualize the clusters
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # iterate over the clusters
    colors = ['b', 'g', 'r', 'k', 'm', 'yellow', 'aqua']
    for i in range(len(final_kmeans_clusters)):
        x = df_arr[final_kmeans_clusters[i]][:, 0]
        y = df_arr[final_kmeans_clusters[i]][:, 1]
        z = df_arr[final_kmeans_clusters[i]][:, 2]
        ax.scatter3D(x, y, z, color=colors[i]) # plot the i-th cluster
    plt.title("k-means clustering results")
    plt.savefig("kmeans_clusters_plot.png")


    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
