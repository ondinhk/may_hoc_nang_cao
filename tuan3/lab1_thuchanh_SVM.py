import numpy as np
import matplotlib.pyplot as plot
import quadprog as qp

X = np.matrix([[2, 2],
               [3, 1],
               [1, 1]])

Y = np.array([1, 1, -1])

# Ma Tran G
G = np.matrix([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0000001]
               ])
a = np.array(([0.0, 0.0, 0.0]))

# Ma Tran C
C = np.matrix([
    [2.0, 2.0, -1.0],
    [3.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0]
]).T
b = np.array([1.0, 1.0, 1.0])
# Giai bai toan
sol = qp.solve_qp(G, a, C, b)
# Ket qua
wb = sol[0]
# print(wb)
# Ve
plot.plot(
    X[0:2, 0], X[0:2, 1], 'bo'
)
plot.plot(
    X[2:3, 0], X[2:3, 1], 'rx'
)

x1 = np.array([0, 5])
# x1 + x2 - b = 0
# wb[0] = w1
# wb[1] = w2
# wb[2] = b
a1 = (wb[2] - wb[0] * x1) / wb[1]
# x1 + x2 - b - 1 = 0
a2 = ((wb[2] + 1) - wb[0] * x1) / wb[1]
# x1 + x2 - b +1 = 0
a3 = ((wb[2] - 1) - wb[0] * x1) / wb[1]
plot.plot(x1, a1)
plot.plot(x1, a2)
plot.plot(x1, a3)
#
#
# def duDoan(x, y, wb):
#     result = x * wb[0] + y * wb[1] - wb[2]
#     print(result)
#     if result > 0:
#         print('Lop duong')
#         plot.plot(x, y, 'bo')
#     elif result < 0:
#         print('Lop Am')
#         plot.plot(x, y, 'rx')
#
#
# duDoan(0.5, 2, wb)
plot.show()
