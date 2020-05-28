import matplotlib.pyplot as plt
import numpy as np


emb_path = 'data/sanfrancisco_combined_pca_traffic_labeled_multi_tsne.embeddings'
x_location = (-80, 65)
y_location = (-80, 80)

sf_X = []
sf_Y = []
sf_T = []

with open(emb_path) as f:

    for line in f:
        x_y_label = line.strip().split(' ')
        x = float(x_y_label[0])
        y = float(x_y_label[1])
        label = float(x_y_label[-1])
        sf_X.append(x)
        sf_Y.append(y)
        sf_T.append(label)  # yellow

sf_X = np.array(sf_X)
sf_Y = np.array(sf_Y)
sf_T = np.array(sf_T)

# 散点图
plt.scatter(sf_X, sf_Y, s=5, c=sf_T, alpha=.5)

plt.xlim(x_location[0], x_location[1])
plt.xticks(())  # ignore xticks
plt.ylim(y_location[0], y_location[1])
plt.yticks(())  # ignore yticks

plt.show()

