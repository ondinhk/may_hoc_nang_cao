#
# B1812346 - OnDinhKhang
#
#
import quadprog as qp
import numpy as np
import matplotlib.pyplot as plt


def RBF_kernel(xi, xj, gama=1.0):
    diff = xi - xj
    dist = np.dot(diff, diff.T)
    return np.exp(-gama * dist)


def main():
    c = 10
    X = np.array([[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1],
                  [2, 0],
                  [0, 2],
                  [-2, 0],
                  [0, -2],
                  [0, -3]])
    Y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1])
    for idx, value in enumerate(Y):
        if value == 1:
            plt.plot(
                X[idx, 0:1], X[idx, 1:2], 'bo'
            )
        else:
            plt.plot(
                X[idx, 0:1], X[idx, 1:2], 'rx'
            )
    kernel = RBF_kernel
    # Tạo một ma trận có đường chéo là Y
    D = np.diag(Y)
    m = len(Y)
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i][j] = kernel(X[i], X[j])
    G = np.dot(np.dot(D, K), D)
    G = 0.5 * (G + G.T)
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
    # tìm W xong tính b, cuối cùng vẽ.
    w = sum(np.dot(np.diag(alpha), np.dot(D, X)), 0).T
    k = np.argmax(alpha)
    w1 = w[0]
    w2 = w[1]
    #
    # # bias
    # b = np.dot(X[k], w) - Y[k]
    # print(b)
    for k in range(len(alpha)):
        if alpha[k] < c:
            b = 0
            for i in range(m):
                b = b + alpha[i] * Y[i] * kernel(X[i], X[k])
            # break
    # Ve duong thang
    x1 = np.array([0, 3])
    d1 = (b - w1 * x1) / w2
    # x1 + x2 - b = -1
    d2 = ((b - 1) - w1 * x1) / w2
    # x1 + x2 - b = 1
    d3 = ((b + 1) - w1 * x1) / w2

    plt.plot(x1, d1)
    plt.plot(x1, d2)
    plt.plot(x1, d3)
    plt.show()


if __name__ == "__main__":
    main()
