import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

ax = np.array([[4, 90], [15, 50], [65, 70], [5, 7], [18, 3], [70, 80]], dtype=np.float32)
ay = np.array([1, 2, 0, 1, 2, 0], dtype=np.float32)
qx = np.array([[20, 6], [16, 80], [68, 8], [80, 20], [7, 70], [3, 50]], dtype=np.float32)
qy = np.array([2, 2, 0, 0, 1, 1], dtype=np.float32)

arvore = tree.DecisionTreeClassifier()
arvore = arvore.fit(ax, ay)
qp = arvore.predict(qx)
print("qp: ", qp)
print("qy: ", qy)

fig = plt.figure(figsize=(8, 6))
tree.plot_tree(arvore, filled=True, fontsize=10)
plt.tight_layout()
plt.show()
fig.savefig("dt2-pele.png")
