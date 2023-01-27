import matplotlib.pyplot as plt
import numpy as np

xx = np.array(
    [
        [1, 2],
        [2, 3]
    ]
)

yy = np.array(
    [
        [1, 1],
        [2, 2]
    ]
)

zz = np.array(
    [
        [1, 2],
        [3, 4]
    ]
)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xx, yy, zz, s=10, c="cyan", label="Data")
ax.plot_surface(
    xx, yy, zz, rstride=1, cstride=1, cmap=plt.cm.jet,
    linewidth=0, antialiased=False, alpha=.5, label="Poly2dFitter"
)
# ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.tight_layout()
