import csv
import numpy as np
import matplotlib.pyplot as plt


def create_csv(filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 't'])
        writer.writerow([1000, 0.2])
        writer.writerow([2000, 0.45])
        writer.writerow([5000, 1.5])
        writer.writerow([10000, 3.2])
        writer.writerow([20000, 7.5])


def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_polynomial(coef, x_data, x):
    n = len(x_data)
    p = coef[0]
    w = 1.0
    for k in range(1, n):
        w *= (x - x_data[k - 1])
        p += w * coef[k]
    return p


def lagrange_polynomial(x_data, y_data, x):
    n = len(x_data)
    p = 0.0
    for i in range(n):
        l = 1.0
        for j in range(n):
            if i != j:
                l *= (x - x_data[j]) / (x_data[i] - x_data[j])
        p += y_data[i] * l
    return p


def w_n_function(x_data, x):
    w = 1.0
    for xi in x_data:
        w *= (x - xi)
    return w


def baseline_function(x):
    return 4.76e-5 * np.power(x, 1.209)


def plot_three_panels(title, x_plot, y_true, y_pred_newton, y_pred_lagrange, x_nodes, y_nodes, w_plot):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].plot(x_plot, y_true, '--k', label='f(x) (baseline)')
    axes[0].plot(x_plot, y_pred_newton, '-b', linewidth=4, alpha=0.5, label='N(x) (Newton)')
    axes[0].plot(x_plot, y_pred_lagrange, '--', color='orange', label='L(x) (Lagrange)')
    axes[0].scatter(x_nodes, y_nodes, color='red', zorder=5, label='Nodes')
    axes[0].set_title('Function and Interpolation')
    axes[0].set_xlabel('Розмір (кількість завдань)')
    axes[0].set_ylabel('Вартість ($)')
    axes[0].grid(True)
    axes[0].legend()

    error = np.abs(np.array(y_true) - np.array(y_pred_newton))
    axes[1].plot(x_plot, error, color='orange', label='e(x) = |f(x) - N(x)|')
    axes[1].set_title('Absolute Error e(x)')
    axes[1].set_xlabel('Розмір (кількість завдань)')
    axes[1].set_ylabel('Абсолютна похибка ($)')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x_plot, w_plot, color='green', label='w_n(x)')
    axes[2].set_title('w_n(x)')
    axes[2].set_xlabel('Розмір (кількість завдань)')
    axes[2].set_ylabel('Значення полінома w_n(x)')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    filename = "data_var4.csv"
    create_csv(filename)
    x_csv, y_csv = read_data(filename)

    coef_csv = divided_differences(x_csv, y_csv)
    target_x = 15000
    forecast_newton = newton_polynomial(coef_csv, x_csv, target_x)
    forecast_lagrange = lagrange_polynomial(x_csv, y_csv, target_x)

    x_plot = np.linspace(min(x_csv), max(x_csv), 200)
    y_true = baseline_function(x_plot)
    y_pred_newton = [newton_polynomial(coef_csv, x_csv, xi) for xi in x_plot]
    y_pred_lagrange = [lagrange_polynomial(x_csv, y_csv, xi) for xi in x_plot]
    w_plot = [w_n_function(x_csv, xi) for xi in x_plot]
    plot_three_panels("Base Model (from CSV data)", x_plot, y_true, y_pred_newton, y_pred_lagrange, x_csv, y_csv,
                      w_plot)

    a_fixed, b_fixed = 1000, 20000
    for n in [5, 10, 20]:
        x_nodes = np.linspace(a_fixed, b_fixed, n)
        y_nodes = baseline_function(x_nodes)
        coef = divided_differences(x_nodes, y_nodes)

        x_plot = np.linspace(a_fixed, b_fixed, 400)
        y_true = baseline_function(x_plot)
        y_pred_newton = [newton_polynomial(coef, x_nodes, xi) for xi in x_plot]
        y_pred_lagrange = [lagrange_polynomial(x_nodes, y_nodes, xi) for xi in x_plot]
        w_plot = [w_n_function(x_nodes, xi) for xi in x_plot]

        plot_three_panels(f"Fixed Interval [{a_fixed}, {b_fixed}], Nodes n={n}",
                          x_plot, y_true, y_pred_newton, y_pred_lagrange, x_nodes, y_nodes, w_plot)

    h_step = 1000
    a_start = 1000
    for n in [5, 10, 20]:
        b_end = a_start + h_step * (n - 1)
        x_nodes = np.linspace(a_start, b_end, n)
        y_nodes = baseline_function(x_nodes)
        coef = divided_differences(x_nodes, y_nodes)

        x_plot = np.linspace(a_start, b_end, 400)
        y_true = baseline_function(x_plot)
        y_pred_newton = [newton_polynomial(coef, x_nodes, xi) for xi in x_plot]
        y_pred_lagrange = [lagrange_polynomial(x_nodes, y_nodes, xi) for xi in x_plot]
        w_plot = [w_n_function(x_nodes, xi) for xi in x_plot]

        plot_three_panels(f"Fixed h={h_step}, Interval [{a_start}, {b_end}], n={n}",
                          x_plot, y_true, y_pred_newton, y_pred_lagrange, x_nodes, y_nodes, w_plot)

    plt.figure(figsize=(12, 7))
    a_runge, b_runge = 1000, 80000
    x_plot_runge = np.linspace(a_runge, b_runge, 400)
    y_true_runge = baseline_function(x_plot_runge)
    plt.plot(x_plot_runge, y_true_runge, '--k', linewidth=2, label='f(x) (baseline)')

    colors = ['blue', 'green', 'red']
    for i, n in enumerate([10, 20, 30]):
        x_nodes = np.linspace(a_runge, b_runge, n)
        y_nodes = baseline_function(x_nodes)
        coef = divided_differences(x_nodes, y_nodes)
        y_pred_runge = [newton_polynomial(coef, x_nodes, xi) for xi in x_plot_runge]
        plt.plot(x_plot_runge, y_pred_runge, color=colors[i], label=f'N(x), n={n}')

    plt.title('Аналіз ефекту Рунге (Штучно розширений інтервал до 80000)', fontsize=14)
    plt.xlabel('Розмір (кількість завдань)')
    plt.ylabel('Вартість ($)')
    plt.grid(True)
    plt.legend()
    plt.ylim(min(y_true_runge) - 5, max(y_true_runge) + 20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()