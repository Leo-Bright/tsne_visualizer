import matplotlib.pyplot as plt
import numpy as np

n = 1024    # data size
X = np.random.normal(0, 1, n) # 每一个点的X值
Y = np.random.normal(0, 1, n) # 每一个点的Y值
T = np.arctan2(Y,X) # for color value

# 散点图
plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-2, 2)
plt.xticks(())  # ignore xticks
plt.ylim(-2, 2)
plt.yticks(())  # ignore yticks

plt.show()

