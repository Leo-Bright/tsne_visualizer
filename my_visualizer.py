import matplotlib.pyplot as plt
import numpy as np


emb_path = 'data/sf_gcn_raw_feature_none_16d_labeled_tsne.embedding'
x_location = (-80, 65)
y_location = (-80, 80)
positive_label = 'normal'

sf_X = []
sf_Y = []
sf_T = []
sf_V = []

with open(emb_path) as f:
    normal_count = 0
    unnormal_count = 0
    for line in f:
        x_y_label = line.strip().split(' ')
        x = float(x_y_label[0])
        y = float(x_y_label[1])
        label = x_y_label[-1]
        if label == positive_label:
            sf_X.append(x)
            sf_Y.append(y)
            sf_T.append(0.5)  # yellow
            sf_V.append(y / x)
            normal_count += 1
        elif unnormal_count < normal_count + 1:
            sf_X.append(x)
            sf_Y.append(y)
            sf_T.append(-0.5)  # purple
            sf_V.append(y / x)
            unnormal_count += 1
        else:
            continue
    print(normal_count, unnormal_count)

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

