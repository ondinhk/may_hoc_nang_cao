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
               [2.5, 1.4]
               ])
Y = np.array([1, 1, -1, -1])
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
# print(m)
# Tạo một ma trận có đường chéo là Y
D = np.diag(Y)
# [[ 1  0  0  0]
#  [ 0  1  0  0]
#  [ 0  0 -1  0]
#  [ 0  0  0 -1]]
# Nhân X với chuyển vị của X
# (X * X.T)
XXT = np.dot(X, X.T)
# len(Y)
m = len(Y)
# Ma tran G = D*(X*X.T)*D
G = np.dot(np.dot(D, XXT), D)
G = 0.5*(G + G.T)
G = G + np.diag(m * [1e-10])
# Vector toan so 1
a = np.array(m * [1.0])
# Ma tran co duong cheo = 1
I = np.diag(m * [1.0])
# ghep rang buoc
C = np.vstack([Y, I, -I]).T
# vector b ma tran 0
c = 1000
# c lớn thì lỗi nhỏ, lề nhỏ
# c nhỏ thì lỗi nhiều, lề lớn
b = np.array((m + 1) * [0.0] + m * [-c])
#
meq = 1
sol = qp.solve_qp(G, a, C, b, meq)
alpha = sol[0]
print(alpha)
# tìm W xong tính b, cuối cùng vẽ.
w = sum(np.dot(np.diag(alpha), np.dot(D, X)), 0).T
k = np.argmax(alpha)
# print(k)
# print(X[k])
#
# # bias
b = np.dot(X[k], w) - Y[k]
b = b[0, 0]
#
# Ve duong thang
x1 = np.array([0, 3])
x2 = (b - w[0, 0] * x1) / w[1, 0]
# x1 + x2 - b = -1
x3 = ((b-1) - w[0, 0] * x1) / w[1, 0]
# x1 + x2 - b = 1
x4 = ((b+1) - w[0, 0] * x1) / w[1, 0]
plt.plot(x1, x2)
plt.plot(x1, x3)
plt.plot(x1, x4)
plt.show()