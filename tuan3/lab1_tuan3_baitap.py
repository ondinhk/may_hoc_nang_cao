import numpy as np
import pandas as pd
import quadprog as qp
import matplotlib.pyplot as plt

data = pd.read_csv('input1.csv', index_col=0)
# ma tran X, Y
X = data.iloc[:, 0:2]
Y = data.iloc[:, 2]
matrix_x = np.matrix(X)
matrix_y = np.matrix(Y)

# Ve hinh
for idx, value in enumerate(Y):
    if value == 1:
        plt.plot(
            matrix_x[idx, 0:1], matrix_x[idx, 1:2], 'bo'
        )
    else:
        plt.plot(
            matrix_x[idx, 0:1], matrix_x[idx, 1:2], 'rx'
        )

# them -1 vao cuoi ma tran x
matrix_x = np.insert(matrix_x, 2, -1, axis=1)
# nhan ma tran x voi y (quadprog)
for idx, item in enumerate(matrix_x):
    matrix_x[idx] = item.dot(matrix_y[0, idx])
# Ma tran C = -G (.T ma tran chuyen vi)
C = matrix_x.astype(np.float_)
C = C.T

# b = -h (=1)
# tao array b voi so phan tu bang voi length cua matrix x
b = np.full(
    shape=len(matrix_x),
    fill_value=1,
    dtype=np.double
)
# a = -q
a = np.array(([0.0, 0.0, 0.0]))
# Ma Tran G
G = np.matrix([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0000001]
               ])

sol = qp.solve_qp(G, a, C, b)
# Ket qua
x1 = np.array([0, 4])
wb = sol[0]
print(wb)
# [ 4.28571429  2.85714286 16.14285714]

# x1 + x2 - b = 0
a1 = (wb[2] - wb[0] * x1) / wb[1]
# x1 + x2 - b = 1
a2 = ((wb[2] - 1) - wb[0] * x1) / wb[1]
# x1 + x2 - b = -1
a3 = ((wb[2] + 1) - wb[0] * x1) / wb[1]
plt.plot(x1, a2)
plt.plot(x1, a3)
plt.plot(x1, a1)
plt.show()
