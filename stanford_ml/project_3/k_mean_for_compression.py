from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

image_matrix_l = imread('../data/project_3/mandrill-large.tiff')
image_matrix_s = imread('../data/project_3/mandrill-small.tiff')

#%%
# plt.imshow(image_matrix_l)
# plt.imshow(image_matrix_s)
# print(image_matrix_s)
# plt.show()
#%%
# IMPORTANt sklearn kmean expect input to be in the shape of (x,n) where n is the dimension and x is the num of data
l_shape = image_matrix_l.shape
s_shape = image_matrix_s.shape
image_matrix_s = image_matrix_s.reshape(-1,3)
image_matrix_l = image_matrix_l.reshape(-1,3)
#%%

if __name__ == "__main__":
    kmean = KMeans(n_clusters=3, n_init=10, max_iter=30, algorithm="full")
    kmean.fit(image_matrix_s)
    clusters = kmean.cluster_centers_
    # IMPORTANT map can only iterate over list
    clusters = np.array(list(map(int, clusters.reshape(-1)))).reshape(-1,3)
    # np.repeat default is axis=1
    color_grid = np.repeat(clusters, clusters.shape[0], axis=0).reshape(clusters.shape[0],-1,3)
    # plt.imshow(color_grid)
    # plt.show()

    prediction = kmean.predict(image_matrix_l)
    prediction = np.array([clusters[num] for num in prediction])
    # IMPORATNT plt.imshow can only show int colors
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(color_grid)
    axes[1].imshow(image_matrix_l.reshape(l_shape))
    axes[2].imshow(prediction.reshape(l_shape))
    plt.show()
    