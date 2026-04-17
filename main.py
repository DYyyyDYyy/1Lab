import numpy as np


try:
    HIGH_PRECISION = np.float128
except AttributeError:
    HIGH_PRECISION = np.longdouble


def write_matrix_to_file(filename, matrix):
    with open(filename, 'w') as f:
        for row in matrix:

            f.write(" ".join(f"{val:.18f}" for val in row) + "\n")


def write_vector_to_file(filename, vector):
    with open(filename, 'w') as f:
        for val in vector:
            f.write(f"{val:.18f}\n")


def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            matrix.append([float(x) for x in line.split()])
    return np.array(matrix, dtype=HIGH_PRECISION)


def read_vector_from_file(filename):
    vector = []
    with open(filename, 'r') as f:
        for line in f:
            vector.append(float(line.strip()))
    return np.array(vector, dtype=HIGH_PRECISION)


def mat_vec_mult(A, X):
    n = len(A)
    B = np.zeros(n, dtype=HIGH_PRECISION)
    for i in range(n):
        B[i] = sum(A[i, j] * X[j] for j in range(n))
    return B


def vector_norm(V):
    return np.max(np.abs(V))


def vector_sub(V1, V2):
    return V1 - V2


def vector_add(V1, V2):
    return V1 + V2


def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n), dtype=HIGH_PRECISION)
    U = np.zeros((n, n), dtype=HIGH_PRECISION)

    for i in range(n):
        U[i, i] = 1.0

    for k in range(n):
        for i in range(k, n):
            sum_val = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - sum_val

        for i in range(k + 1, n):
            sum_val = sum(L[k, j] * U[j, i] for j in range(k))
            U[k, i] = (A[k, i] - sum_val) / L[k, k]

    return L, U


def solve_lu(L, U, B):
    n = len(B)
    Z = np.zeros(n, dtype=HIGH_PRECISION)
    X = np.zeros(n, dtype=HIGH_PRECISION)

    Z[0] = B[0] / L[0, 0]
    for k in range(1, n):
        sum_val = sum(L[k, j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_val) / L[k, k]

    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        sum_val = sum(U[k, j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_val

    return X



n = 100

# Генеруємо матрицю одразу з типом HIGH_PRECISION
A_initial = np.random.uniform(-10.0, 10.0, (n, n)).astype(HIGH_PRECISION)
for i in range(n):
    A_initial[i, i] += HIGH_PRECISION(150.0)  # Запобіжник для стабільності

write_matrix_to_file("matrix_A.txt", A_initial)

# Формуємо точний вектор розв'язку x_j = 2.5
X_exact = np.full(n, 2.5, dtype=HIGH_PRECISION)
B_original = mat_vec_mult(A_initial, X_exact)

write_vector_to_file("vector_B.txt", B_original)


A = read_matrix_from_file("matrix_A.txt")
B = read_vector_from_file("vector_B.txt")

L, U = lu_decomposition(A)
write_matrix_to_file("matrix_L.txt", L)
write_matrix_to_file("matrix_U.txt", U)


X0 = solve_lu(L, U, B)

B_calc = mat_vec_mult(A, X0)
residual = vector_sub(B_calc, B)
eps_initial = vector_norm(residual)
print(f"Початкова похибка (eps) після LU-розкладу: {eps_initial:.4e}")


eps0 = HIGH_PRECISION(1e-14)
iterations = 0
max_iterations = 500
X_current = X0.copy()

while iterations < max_iterations:
    iterations += 1


    B0 = mat_vec_mult(A, X_current)
    R = vector_sub(B, B0)

    delta_X = solve_lu(L, U, R)


    X_next = vector_add(X_current, delta_X)


    norm_delta_X = vector_norm(delta_X)
    norm_residual = vector_norm(vector_sub(mat_vec_mult(A, X_next), B))

    if norm_delta_X <= eps0 and norm_residual <= eps0:
        X_current = X_next
        break

    X_current = X_next

print(f"Кількість ітерацій для досягнення точності: {iterations}")
print(f"Кінцева похибка (нев'язка) після уточнення: {norm_residual:.4e}")