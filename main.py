import csv
import math
import matplotlib.pyplot as plt
import numpy as np

def create_sample_csv(filename='data.csv'):
    data = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Month', 'Temp'])
        writer.writerows(data)

def read_data(filename='data.csv'):
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

def form_matrix(x, m):
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi ** (i + j) for xi in x)
    return A

def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(yi * (xi ** i) for xi, yi in zip(x, y))
    return b

def gauss_solve(A, b):
    n = len(b)
    A_copy = [row[:] for row in A]
    b_copy = b[:]

    # Прямий хід з вибором головного елемента
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A_copy[i][k]) > abs(A_copy[max_row][k]):
                max_row = i
        A_copy[k], A_copy[max_row] = A_copy[max_row], A_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        for i in range(k + 1, n):
            if A_copy[k][k] == 0:
                continue
            factor = A_copy[i][k] / A_copy[k][k]
            for j in range(k, n):
                A_copy[i][j] -= factor * A_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    # Зворотний хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        if A_copy[i][i] != 0:
            x_sol[i] = (b_copy[i] - s) / A_copy[i][i]
        else:
            x_sol[i] = 0.0
    return x_sol

def evaluate_poly(x_list, coef):
    return [sum(c * (xi ** i) for i, c in enumerate(coef)) for xi in x_list]

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((yt - ya) ** 2 for yt, ya in zip(y_true, y_approx)) / n

def plot_all_errors(x, y, coefs_to_plot, n):
    x0, xn = x[0], x[-1]
    h1 = (xn - x0) / (20 * n)
    x_fine = np.arange(x0, xn + h1, h1)
    y_fine_true = np.interp(x_fine, x, y)

    num_plots = len(coefs_to_plot)


    plt.figure(num="Вікно 2: Табуляція похибок", figsize=(12, 7))
    plt.title(f"Табуляція похибки ε(x) для m = 1...{num_plots}", fontsize=16)


    for m_idx in range(num_plots):
        m = m_idx + 1
        coef = coefs_to_plot[m_idx]
        y_fine_approx = evaluate_poly(x_fine, coef)
        error_fine = [abs(yt - ya) for yt, ya in zip(y_fine_true, y_fine_approx)]

        plt.plot(x_fine, error_fine, label=f"m={m}")

    plt.xlabel("Місяць")
    plt.ylabel("Похибка")
    plt.grid(True)
    plt.legend() # Додаємо легенду
    plt.tight_layout()

def main():
    create_sample_csv()
    x, y = read_data()
    n_nodes = len(x)

    max_degree = 10
    limit_m = 10 # <--- Обмеження

    variances = []
    all_coefs = []

    best_m = 1
    min_var = float('inf')
    best_coef = []

    print("--- Дисперсії для різних степенів полінома ---")
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b = form_vector(x, y, m)
        coef = gauss_solve(A, b)
        all_coefs.append(coef)

        y_approx = evaluate_poly(x, coef)
        var = variance(y, y_approx)
        variances.append(var)
        print(f"Степінь m = {m:2d} | Дисперсія = {var:.4f}")

        if var < min_var and m <= limit_m:
            min_var = var
            best_m = m
            best_coef = coef

    print(f"\n=> Для безпечної апроксимації та прогнозу обрано степінь: m = {best_m}")

    y_opt_approx = evaluate_poly(x, best_coef)
    x_future = [25, 26, 27]
    y_future = evaluate_poly(x_future, best_coef)

    print("\n--- Прогноз на наступні 3 місяці ---")
    for xf, yf in zip(x_future, y_future):
        print(f"Місяць {xf}: {yf:.2f} градусів")

    errors = [abs(yt - ya) for yt, ya in zip(y, y_opt_approx)]

    plt.figure(num="Вікно 1: Основні результати", figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, max_degree + 1), variances, marker='o', color='purple')
    plt.title("Залежність дисперсії від степеня m")
    plt.xlabel("Степінь полінома (m)")
    plt.ylabel("Дисперсія")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x, y, 'ro', label="Фактичні дані")
    plt.plot(x, y_opt_approx, 'b-', label=f"Апроксимація (m={best_m})")
    plt.plot(x_future, y_future, 'gP', markersize=10, label="Прогноз")
    plt.title("Апроксимація та екстраполяція")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, errors, 'r-', marker='x')
    plt.title("Похибка апроксимації ε(x) у вузлах")
    plt.xlabel("Місяць")
    plt.ylabel("Абсолютна похибка")
    plt.grid(True)
    plt.tight_layout()

    plot_all_errors(x, y, all_coefs[:limit_m], n_nodes)

    plt.show()

if __name__ == "__main__":
    main()