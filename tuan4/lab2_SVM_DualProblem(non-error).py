#
# B1812346 - OnDinhKhang
#
#
import numpy as np
import matplotlib.pyplot as plt
import quadprog as qp

X = np.matrix([[2.0, 2],
               [3, 1],
               [1, 1],
               ])
Y = np.array([1, 1, -1])
# Ve
for idx, value in enumerate(Y):
    if value == 1:
        plt.plot(
            X[idx, 0:1], X[idx, 1:2], 'bo'
        )
    else:
        plt.plot(
            X[idx, 0:1], X[idx, 1:2], 'rx'
        )

# Tạo một ma trận có đường chéo là Y
D = np.diag(Y)

m = len(Y)
# [[ 1  0  0]
#  [ 0  1  0]
#  [ 0  0 -1]]

# Nhân X với chuyển vị của X
# (X * X.T)
XXT = np.dot(X, X.T)
# Ma tran G = D*(X*X.T)*D

G = np.dot(np.dot(D, XXT), D)

G = G + np.diag(m * [1e-10])
# Vector toan so 1
a = np.array(m * [1.0])
# Ma tran co duong cheo = 1
I = np.diag(m * [1.0])
# ghep rang buoc
C = np.vstack([Y, I]).T
# vector b ma tran 0
b = np.array((m + 1) * [0.0])
meq = 1

sol = qp.solve_qp(G, a, C, b, meq)
alpha = sol[0]
# print(alpha)
# tìm W xong tính b, cuối cùng vẽ.
w = sum(np.dot(np.diag(alpha), np.dot(D, X)), 0).T
print(w[0, 0])
print(w[1, 0])

k = np.argmax(alpha)
# bias
b = np.dot(X[k], w) - Y[k]
b = b[0, 0]

# Ve duong thang
x1 = np.array([0, 3])
d1 = (b - w[0, 0] * x1) / w[1, 0]
# x1 + x2 - b = -1
d2 = ((b - 1) - w[0, 0] * x1) / w[1, 0]
# x1 + x2 - b = 1
d3 = ((b + 1) - w[0, 0] * x1) / w[1, 0]

plt.plot(x1, d1)
plt.plot(x1, d2)
plt.plot(x1, d3)
plt.show()
