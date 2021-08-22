# Coronavirus-Data-Clustering
Clusters Coronavirus data using k-means clustering.
The data is clustered into groups where the viruses are having similar characteristics.


### DATASET:
	-> The dataset contains Covid-19 spread characteristics for different regions worldwide
	-> Attributes: mortality rate, transmission rate and incubation period 


### RUNNING THE PROGRAM:
	The program is written in Python 3.8.5. To execute it, enter the below command in terminal:
	python k_means.py
	
	-> The './data/COVID_4_unlabelled.csv' file should be present at the current working directory before executing the program.
	-> When executed, the code saves the cluster informations for k-means clustering in the file 'kmeans_clusters.txt' in the current working directory.
	-> The other required informations are printed in the console.
	-> The time taken by the program to run all the steps is approximately 5 seconds.


### RESULTS:
	k-means clustering is done with 5 different values of k and the silhouette coefficients are calculated.
		k = 3 : silhouette coefficient = 0.863
		k = 4 : silhouette coefficient = 0.907
		k = 5 : silhouette coefficient = 0.914
		k = 6 : silhouette coefficient = 0.907
		k = 7 : silhouette coefficient = 0.898
	Highest silhouette coefficient = 0.914, obtained for k = 5
	

### VISUALIZATION:
	-> As the dataset has 3 attributes, we can visualize it using 3D plots.
	-> A 3D plot of the clusters created is stored as 'kmeans_clusters_plot.png' in the current working directory after successful execution.
	-> A 3D plot of the original dataset is stored as 'original_dataset_plot.png' in the current working directory after successful execution.
	
![alt text](https://github.com/soumyaporel/Coronavirus-Data-Clustering/blob/main/plots/original_dataset_plot.png?raw=true)
![alt text](https://github.com/soumyaporel/Coronavirus-Data-Clustering/blob/main/plots/kmeans_clusters_plot.png?raw=true)
