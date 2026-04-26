import numpy as np


def generate_matrix(n):
    A = np.random.uniform(-10, 10, (n, n)).astype(np.float128)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 10)
    return A


def simple_iteration(A, B, x0, eps):
    norm_A = np.linalg.norm(A, np.inf)
    tau = np.float128(1.0) / norm_A
    x_k = np.copy(x0)
    iters = 0
    while True:
        residual = np.dot(A, x_k) - B
        err = np.linalg.norm(residual)
        if err <= eps:
            break
        x_k = x_k - tau * residual
        iters += 1
    return x_k, iters, err


def jacobi(A, B, x0, eps):
    D = np.diag(np.diag(A))
    LU = A - D
    x_k = np.copy(x0)
    iters = 0
    D_inv = np.diag(np.float128(1.0) / np.diag(D))
    while True:
        x_new = np.dot(D_inv, B - np.dot(LU, x_k))
        err = np.linalg.norm(x_new - x_k)
        if err < eps:
            break
        x_k = x_new
        iters += 1
    return x_new, iters, err


def gauss_seidel(A, B, x0, eps):
    n = len(A)
    x_k = np.copy(x0)
    iters = 0
    while True:
        x_old = np.copy(x_k)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_k[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x_k[i] = (B[i] - s1 - s2) / A[i, i]
        err = np.linalg.norm(x_k - x_old)
        if err < eps:
            break
        iters += 1
    return x_k, iters, err


n = 100
eps = np.float128(1e-14)

A_gen = generate_matrix(n)
np.savetxt('matrix_A.txt', A_gen)

x_exact = np.full(n, 2.5, dtype=np.float128)
B_gen = np.dot(A_gen, x_exact)
np.savetxt('vector_B.txt', B_gen)

A = np.loadtxt('matrix_A.txt', dtype=np.float128)
B = np.loadtxt('vector_B.txt', dtype=np.float128)

x0 = np.full(n, 1.0, dtype=np.float128)

x_simple, iters_simple, err_simple = simple_iteration(A, B, x0, eps)
x_jacobi, iters_jacobi, err_jacobi = jacobi(A, B, x0, eps)
x_seidel, iters_seidel, err_seidel = gauss_seidel(A, B, x0, eps)

with open('results_X.txt', 'w') as f:
    f.write("Simple Iteration Results:\n")
    np.savetxt(f, x_simple, fmt='%.30e')
    f.write(f"Iterations: {iters_simple}, Error: {err_simple}\n\n")

    f.write("Jacobi Results:\n")
    np.savetxt(f, x_jacobi, fmt='%.30e')
    f.write(f"Iterations: {iters_jacobi}, Error: {err_jacobi}\n\n")

    f.write("Gauss-Seidel Results:\n")
    np.savetxt(f, x_seidel, fmt='%.30e')
    f.write(f"Iterations: {iters_seidel}, Error: {err_seidel}\n")

print(f"Метод простої ітерації: {iters_simple} ітерацій, похибка: {err_simple}")
print(f"Метод Якобі: {iters_jacobi} ітерацій, похибка: {err_jacobi}")
print(f"Метод Гауса-Зейделя: {iters_seidel} ітерацій, похибка: {err_seidel}")