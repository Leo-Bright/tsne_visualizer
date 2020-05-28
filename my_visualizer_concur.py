import matplotlib.pyplot as plt
import numpy as np


emb_paths = ['data/sanfrancisco_raw_feature_crossing_tsne.embeddings',
            'data/sf_node2vec_128_labeled_tsne.embeddings',
            'data/sf_gcn_raw_feature_none_16d_labeled_tsne.embedding',
            'data/sanfrancisco_shortest_dist_increment_tag_type_distance1000_with_crossing_labeled_tsne.embedding'
            ]

# raw_x_location = (-7, 10)
# raw_y_location = (-23, -5)
#
# node2vec_x_location = (-100, 100)
# nove2vec_y_location = (-100, 100)
#
# gcn_x_location = (-80, 65)
# gcn_y_location = (-80, 76)
#
# mymodel_x_location = (-74, 80)
# mymodel_y_location = (-76, 80)

x_locations = []
y_locations = []

x_locations.append((-7, 10))
y_locations.append((-23, -5))

x_locations.append((-100, 100))
y_locations.append((-100, 100))

x_locations.append((-80, 65))
y_locations.append((-80, 76))

x_locations.append((-74, 80))
y_locations.append((-76, 80))

positive_label = 'normal'

sf_X = []
sf_Y = []
sf_T = []

for emb_path in  emb_paths:

    _X = []
    _Y = []
    _T = []
    normal_count = 0
    unnormal_count = 0

    with open(emb_path) as f:
        for line in f:
            x_y_label = line.strip().split(' ')
            x = float(x_y_label[0])
            y = float(x_y_label[1])
            label = x_y_label[-1]
            if label == positive_label:
                _X.append(x)
                _Y.append(y)
                _T.append(0.5)  # yellow
                normal_count += 1
            elif unnormal_count < normal_count + 1:
                _X.append(x)
                _Y.append(y)
                _T.append(-0.5)  # purple
                unnormal_count += 1
            else:
                continue

    sf_X.append(np.array(_X))
    sf_Y.append(np.array(_Y))
    sf_T.append(np.array(_T))

# 多个散点图
plt.figure()
for i in range(len(sf_X)):
    plt.subplot(2, 2, i+1)
    plt.scatter(sf_X[i], sf_Y[i], s=3, c=sf_T[i], alpha=.4)
    plt.xlim(x_locations[i][0], x_locations[i][1])
    plt.xticks(())  # ignore xticks
    plt.ylim(y_locations[i][0], y_locations[i][1])
    plt.yticks(())  # ignore yticks

plt.show()

