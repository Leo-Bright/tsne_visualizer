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

sf_X = []
sf_Y = []
sf_T = []

colors = {0: color_0,
          1: color_1,
          2: color_2,
          3: color_3,
          4: color_4,
          5: color_5,
          6: color_6,
          7: color_7,
          8: color_8}

with open(emb_path) as f:

    for line in f:
        x_y_label = line.strip().split(' ')
        x = float(x_y_label[0])
        y = float(x_y_label[1])
        label = int(x_y_label[-1])
        sf_X.append(x)
        sf_Y.append(y)
        if label == -1:
            label = 7
        sf_T.append(colors[label])

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

