#1.1.1 ПРИМЕР
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

kmeans = KMeans(n_clusters=10, random_state = 0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)
fig, ax = plt.subplots(2, 5, figsize = (8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

#задание
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

x = np.array([[5,3], [10,15], [15, 12], [24,10], [30,45], [85,70], [71,80], [60,78], [55,52], [80,91],])

plt.scatter(x[:, 0], x[:, 1], s=50)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=200, alpha=0.5);
plt.show()
"""
#задание
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);
plt.show()
"""
#1.1.2
import matplotlib.pyplot as plt
import numpy as np

x = np.array([[7,8], [12,20], [17, 19], [26,15], [32,37], [87,75], [73,85], [62,80], [73,60], [87,96],])
labels = range(1, 11)
plt.figure(figsize= (10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(x[:, 0], x[:, 1], label = 'True Position')
for label, x1, y in zip(labels,x[:, 0], x[:, 1]):
    plt.annotate(
        label, xy=(x1,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(x, 'single')
labellist = range(1,11)
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels = labellist,
           distance_sort='descending', show_leaf_counts= True)
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c = cluster.labels_, cmap='rainbow')
plt.show()

#пример
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/lucko515/clustering-python/master/Customer%20in%20Mall%20clusterng/Mall_Customers.csv'
customer_data = pd.read_csv(url)
print(customer_data.head())
print(customer_data.shape)

data = customer_data.iloc[:, 3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(28, 12), dpi=180)
plt.title("Customer Dendrogranum")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
print(cluster.fit_predict(data))

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c = cluster.labels_, cmap='plasma')
plt.show()


#задание
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets._samples_generator import make_blobs

customer_data = sns.load_dataset('iris')
print(customer_data.head())
print(customer_data.shape)

data = customer_data.iloc[:, 2:4].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(28, 12), dpi=180)
plt.title("Iris of Fisher dendrogranum")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
print(cluster.fit_predict(data))

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c = cluster.labels_, cmap='plasma')
plt.show()"""
