#
# B1812346 - OnDinhKhang
#
#
import numpy as np
import matplotlib.pyplot as plt
import quadprog as qp

X = np.matrix([[2, 2],
               [3, 1],
               [1, 1],
               [2.5, 1.4]])
Y = np.array([1, 1, -1, -1])
# Vẽ
for idx, value in enumerate(Y):
    if value == 1:
        plt.plot(
            X[idx, 0:1], X[idx, 1:2], 'bo'
        )
    else:
        plt.plot(
            X[idx, 0:1], X[idx, 1:2], 'rx'
        )
G = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001]])
# Nếu c nhỏ -> lề lớn
# Nếu c lớn -> lỗi nhỏ lề nhỏ
c = 10
a = np.array([0.0, 0.0, 0.0, -c, -c, -c, -c])
# Ma tran C
C = np.matrix([[2.0, 2.0, -1.0, 1.0, 0.0, 0.0, 0.0],
               [3.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
               [-1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
               [-2.5, -1.4, 1.0, 0.0, 0.0, 0.0, 1.0],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T
# ----------------------
b = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
sol = qp.solve_qp(G, a, C, b)
wb = sol[0]
print(wb)
x1 = np.array([0, 5])
# x1 + x2 - b = 0
a1 = (wb[2] - wb[0] * x1) / wb[1]
plt.plot(x1, a1)

# x1 + x2 - b = 1
a2 = ((wb[2] + 1) - wb[0] * x1) / wb[1]
plt.plot(x1, a2)

# x1 + x2 - b = -1
a3 = ((wb[2] - 1) - wb[0] * x1) / wb[1]
plt.plot(x1, a3)

# Show
plt.show()