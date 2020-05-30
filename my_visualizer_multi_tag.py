import matplotlib.pyplot as plt
import numpy as np


emb_path = 'data/sanfrancisco_combined_pca_traffic_labeled_multi_tsne.embeddings'
x_location = (-80, 65)
y_location = (-80, 80)
color_0 = 'aliceblue'
color_1 = 'brown'
color_2 = 'cyan'
color_3 = 'darkkhaki'
color_4 = 'darkviolet'
color_5 = 'greenyellow'
color_6 = 'lightsalmon'
color_7 = 'mediumblue'
color_8 = 'pink'

sf_X = {}
sf_Y = {}
sf_T = {}

colors = {0: color_0,
          1: color_1,
          2: color_2,
          3: color_3,
          4: color_4,
          5: color_5,
          6: color_6,
          7: color_7,
          8: color_8}

labels = {0: 'turning_loop',
          1: 'give_way',
          2: 'bus_stop',
          3: 'turning_circle',
          4: 'traffic_signals',
          5: 'crossing',
          6: 'stop',
          7: 'others',
          8: 'motorway'}

sf_T = colors

with open(emb_path) as f:

    for line in f:
        x_y_label = line.strip().split(' ')
        x = float(x_y_label[0])
        y = float(x_y_label[1])
        label = int(x_y_label[-1])
        if label == -1:
            label = 7

        if label not in sf_X:
            sf_X[label] = []
        sf_X[label].append(x)

        if label not in sf_Y:
            sf_Y[label] = []
        sf_Y[label].append(y)

# sf_X = np.array(sf_X)
# sf_Y = np.array(sf_Y)
# sf_T = np.array(sf_T)

for label in sf_X:
    sf_X[label] = np.array(sf_X[label])

for label in sf_Y:
    sf_Y[label] = np.array(sf_Y[label])

# 散点图
sca_0 = plt.scatter(sf_X[0], sf_Y[0], s=5, c=sf_T[0], alpha=.5)
sca_1 = plt.scatter(sf_X[1], sf_Y[1], s=5, c=sf_T[1], alpha=.5)
sca_2 = plt.scatter(sf_X[2], sf_Y[2], s=5, c=sf_T[2], alpha=.5)
sca_3 = plt.scatter(sf_X[3], sf_Y[3], s=5, c=sf_T[3], alpha=.5)
sca_4 = plt.scatter(sf_X[4], sf_Y[4], s=5, c=sf_T[4], alpha=.5)
sca_5 = plt.scatter(sf_X[5], sf_Y[5], s=5, c=sf_T[5], alpha=.5)
sca_6 = plt.scatter(sf_X[6], sf_Y[6], s=5, c=sf_T[6], alpha=.5)
sca_7 = plt.scatter(sf_X[7], sf_Y[7], s=5, c=sf_T[7], alpha=.5)
sca_8 = plt.scatter(sf_X[8], sf_Y[8], s=5, c=sf_T[8], alpha=.5)
plt.legend(handles=[sca_0, sca_1, sca_2, sca_3, sca_4, sca_5, sca_6, sca_7, sca_8],
           labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8]],
           loc='upper right')

plt.xlim(x_location[0], x_location[1])
plt.xticks(())  # ignore xticks
plt.ylim(y_location[0], y_location[1])
plt.yticks(())  # ignore yticks

plt.show()

