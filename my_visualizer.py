import matplotlib.pyplot as plt
import numpy as np


sf_X = []
sf_Y = []
sf_T = []

with open('data/sanfrancisco_random_node2vec_d128_wl1280_node_tsne.embeddings') as f:
    normal_count = 0
    unnormal_count = 0
    for line in f:
        x_y_label = line.strip().split(' ')
        x = x_y_label[0]
        y = x_y_label[1]
        label = x_y_label[-1]
        if label == 'normal':
            sf_X.append(x)
            sf_Y.append(y)
            sf_T.append(0.5)
            normal_count += 1
        elif unnormal_count < normal_count + 1:
            sf_X.append(x)
            sf_Y.append(y)
            sf_T.append(-0.5)
            unnormal_count += 1
        else:
            continue
    print(normal_count, unnormal_count)

sf_X = np.array(sf_X)
sf_Y = np.array(sf_Y)
sf_T = np.array(sf_T)

# 散点图
plt.scatter(sf_X, sf_Y, s=5, c=sf_T, alpha=.5)

plt.xlim(-200, 200)
plt.xticks(())  # ignore xticks
plt.ylim(-200, 200)
plt.yticks(())  # ignore yticks

plt.show()

