import csv 


def cast_to_float_2d(string_list_2d):
    float_list_2d = []
    for row in string_list_2d:

        float_row = [float(cell) for cell in row]
        float_list_2d.append(float_row)
    return float_list_2d

def load_to_memory(original_file_path):
    # Initialize an empty list to store the data
    data_2d_list = []

    # Open the CSV file
    with open(original_file_path, 'r', encoding="utf-8-sig", newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile,delimiter=",")
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append each row to the 2D list
            data_2d_list.append(row)

    
    return data_2d_list # Return the resulting 2D list representing the dataset 



original_file_path_merged = 'tasks/Part_B/merged.csv'
dataset_merged=load_to_memory(original_file_path=original_file_path_merged)

# we will remove the frail column (index = 9) as it is the obtained label. the project description does not elaborate on the 5 columns that produce the frail index. 
temp=[]
for i in dataset_merged:
    temp.append(i[0:9]+i[9+1:])
dataset_merged=temp


header=dataset_merged[0]
entries=dataset_merged[1:]
#print(header)

#we now cast to float
entries=cast_to_float_2d(entries)


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Assuming your data is stored in a 2D list named 'data'
# Each row corresponds to a data point, and columns are features
# Ignore the first column (row id)


# Convert the 2D list to a numpy array and exclude the first column since it is a row id 
data = np.array([row[1:] for row in entries])

mean = np.mean(data, axis=0)




# Step 3: Compute the covariance matrix
cov_matrix = np.cov(data/mean, rowvar=False)

# Step 4: Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Choose the top k eigenvectors
k = 12  # Choose the desired dimensionality
top_eigenvectors = eigenvectors[:, :k]

# Step 7: Form a projection matrix
projection_matrix = top_eigenvectors

# Step 8: Project the original dataset onto the new subspace
reduced_data = np.dot(data/mean, projection_matrix)


# Normalize the dataset
normalized_data = (data)/mean  #this way we equalie the weight of each field to the output
X=normalized_data
X=reduced_data

error_list=[]
sil_list=[]
for i in range(2,20):
    # Define the number of clusters (k)
    k = i

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=i, random_state=42,n_init=20,max_iter=500)
    labels = kmeans.fit_predict(X)
    wcss = kmeans.inertia_
    error_list.append(wcss/10000)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, labels)
    sil_list.append(silhouette_avg)
    #print(f"Silhouette Score for {i} clusters: {silhouette_avg}")

import matplotlib.pyplot as plt

# Create a plot with the first list
plt.plot(error_list, label='WCSS/10000')

# Add the second list to the same plot
plt.plot(sil_list, label='Silhouette Index')

# Add labels and a legend
plt.title('Plot of Two Lists')
plt.xlabel('X-axis Cluster Number')
plt.ylabel('Y-axis (Values)')
plt.legend()

# Show the plot
plt.show()

#without pca: from the plot we can visually identify the elbow on n=9 clusters and the corresponding silhoutte index is 0.291

#by experiementing with pca we gain the following:
#components  #elbow  #silhouette value
#2              #2        0.965
#3              #4        0.775
#4              #5        0.778
#5              #6        0.771
#6              #6        0.771
#7              #7        0.697
#8              #9        0.642
#9              #9        0.641
#10             #9        0.564
#11             #9        0.510
#12             #9        0.472

#At this point we conclude with a 9 centroids clustering and a silhouette index of 0.642 with 8 components
