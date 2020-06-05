import matplotlib.pyplot as plt
import numpy as np


labels = ('Ave', 'unnormal')
colors = ('darkviolet', 'brown')
show_legend = False  # 是否开启图注

emb_path = 'data/segment/sanfrancisco_raw_feature_segment_labeled_tsne.embeddings'
x_location = (-30, 15)
y_location = (-30, 10)

sf_X = {}
sf_Y = {}
sf_T = {}

for i in range(len(labels)):
    sf_T[labels[i]] = colors[i]

with open(emb_path) as f:

    for line in f:
        x_y_label = line.strip().split(' ')
        x = float(x_y_label[0])
        y = float(x_y_label[1])
        label = x_y_label[-1]

        if label not in sf_X:
            sf_X[label] = []
        sf_X[label].append(x)

        if label not in sf_Y:
            sf_Y[label] = []
        sf_Y[label].append(y)

for label in sf_X:
    sf_X[label] = np.array(sf_X[label])
    sf_Y[label] = np.array(sf_Y[label])

# 散点图
scatters = []
for label in labels:
    _scatter = plt.scatter(sf_X[label], sf_Y[label], s=5, c=sf_T[label], alpha=.5)
    scatters.append(_scatter)

if show_legend:
    plt.legend(handles=scatters,
               labels=labels,
               loc='upper right')

plt.xlim(x_location[0], x_location[1])
plt.xticks(())  # ignore xticks
plt.ylim(y_location[0], y_location[1])
plt.yticks(())  # ignore yticks

plt.show()

