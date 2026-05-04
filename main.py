import numpy as np
import os


def F(x):
    return np.exp(x) - 3 * x


def dF(x):
    return np.exp(x) - 3


def d2F(x):
    return np.exp(x)


def tabulate_and_find_roots(a, b, h, filename="tabulation.txt"):
    x = np.arange(a, b + h / 2, h)
    y = F(x)
    data = np.column_stack((x, y))
    np.savetxt(filename, data, header="x\t\tF(x)", fmt="%.6f\t%.12e")
    sign = np.sign(y)
    sign_change = np.where(sign[:-1] * sign[1:] < 0)[0]
    roots = []
    for i in sign_change:
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        root_approx = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(root_approx)
    return roots


def simple_iteration(tau, x0, eps=1e-10, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        x_new = x + tau * F(x)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, _ + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод простої ітерації не збігся")


def newton(x0, eps=1e-10, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = F(x)
        dfx = dF(x)
        x_new = x - fx / dfx
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, i + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод Ньютона не збігся")


def chebyshev(x0, eps=1e-10, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        correction = fx / dfx + 0.5 * (fx ** 2) * d2fx / (dfx ** 3)
        x_new = x - correction
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            return x_new, i + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод Чебишева не збігся")


def chord_method(x0, x1, eps=1e-10, max_iter=1000):
    x_prev, x_curr = x0, x1
    for i in range(max_iter):
        f_prev = F(x_prev)
        f_curr = F(x_curr)
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if abs(F(x_new)) < eps and abs(x_new - x_curr) < eps:
            return x_new, i + 1, abs(F(x_new))
        x_prev, x_curr = x_curr, x_new
    raise RuntimeError("Метод хорд не збігся")


def parabola_method(x0, x1, x2, eps=1e-10, max_iter=1000):
    x = [x0, x1, x2]
    for i in range(max_iter):
        f0, f1, f2 = F(x[0]), F(x[1]), F(x[2])
        h1 = x[1] - x[0]
        h2 = x[2] - x[1]
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2
        if a == 0:
            dx = -c / b
        else:
            disc = b * b - 4 * a * c
            sqrt_disc = np.sqrt(disc)
            dx1 = (-b + sqrt_disc) / (2 * a)
            dx2 = (-b - sqrt_disc) / (2 * a)
            dx = dx1 if abs(dx1) < abs(dx2) else dx2
        x_new = x[2] + dx
        if abs(F(x_new)) < eps and abs(dx) < eps:
            return x_new, i + 1, abs(F(x_new))
        x = [x[1], x[2], x_new]
    raise RuntimeError("Метод парабол не збігся")


def inverse_interpolation_3point(x0, x1, x2, eps=1e-10, max_iter=1000):
    x = [x0, x1, x2]
    for i in range(max_iter):
        y0, y1, y2 = F(x[0]), F(x[1]), F(x[2])
        denom1 = (y0 - y1) * (y0 - y2)
        denom2 = (y1 - y0) * (y1 - y2)
        denom3 = (y2 - y0) * (y2 - y1)
        x_new = (y1 * y2 * x[0]) / denom1 + (y0 * y2 * x[1]) / denom2 + (y0 * y1 * x[2]) / denom3
        if abs(F(x_new)) < eps and abs(x_new - x[2]) < eps:
            return x_new, i + 1, abs(F(x_new))
        x = [x[1], x[2], x_new]
    raise RuntimeError("Метод зворотної інтерполяції не збігся")


def read_poly_coeffs(filename="algebraic_coeffs.txt"):
    with open(filename, "r") as f:
        line = f.readline().strip()
    coeffs = np.array(list(map(float, line.split())))
    return coeffs


def horner_value_and_derivative(coeffs, x):
    n = len(coeffs) - 1
    b = np.zeros(n + 1)
    b[n] = coeffs[0]
    for i in range(n - 1, -1, -1):
        b[i] = coeffs[n - i] + x * b[i + 1]
    c = np.zeros(n + 1)
    c[n] = b[n]
    for i in range(n - 1, 0, -1):
        c[i] = b[i] + x * c[i + 1]
    return b[0], c[1]


def horner_eval(coeffs, x):
    val, _ = horner_value_and_derivative(coeffs, x)
    return val


def newton_horner(coeffs, x0, eps=1e-10, max_iter=1000):
    x = x0
    for i in range(max_iter):
        val, deriv = horner_value_and_derivative(coeffs, x)
        x_new = x - val / deriv
        if abs(horner_eval(coeffs, x_new)) < eps and abs(x_new - x) < eps:
            return x_new, i + 1, abs(horner_eval(coeffs, x_new))
        x = x_new
    raise RuntimeError("Метод Ньютона (Горнер) не збігся")


def lin_method(coeffs, alpha0, beta0, eps=1e-10, max_iter=1000):
    n = len(coeffs) - 1
    alpha, beta = alpha0, beta0
    p = -2 * alpha
    q = alpha ** 2 + beta ** 2
    for it in range(max_iter):
        b = np.zeros(n + 1)
        b[n] = coeffs[0]
        b[n - 1] = coeffs[1] - p * b[n]
        for i in range(n - 2, 1, -1):
            b[i] = coeffs[n - i] - p * b[i + 1] - q * b[i + 2]
        b2 = b[2]
        b3 = b[3] if n >= 3 else 0
        a0 = coeffs[n]
        a1 = coeffs[n - 1]
        if abs(b2) < 1e-15:
            raise RuntimeError("b2 близький до нуля, метод Ліна не застосовний")
        q_new = a0 / b2
        p_new = (a1 * b2 - a0 * b3) / (b2 ** 2)
        alpha_new = -p_new / 2
        disc = q_new - alpha_new ** 2
        if disc < 0:
            raise RuntimeError("Від'ємний дискримінант у методі Ліна")
        beta_new = np.sqrt(disc)
        if abs(alpha_new - alpha) < eps and abs(beta_new - beta) < eps:
            return alpha_new, beta_new, it + 1
        alpha, beta = alpha_new, beta_new
        p, q = p_new, q_new
    raise RuntimeError("Метод Ліна не збігся")


if __name__ == "__main__":
    a, b = 0.0, 2.0
    h = 0.1
    eps = 1e-10
    roots_approx = tabulate_and_find_roots(a, b, h)
    if len(roots_approx) < 2:
        raise SystemExit("Не знайдено двох коренів!")
    root1 = roots_approx[0]
    root2 = roots_approx[1]
    if dF(root1) > 0:
        root1, root2 = root2, root1
    tau1 = -0.5 / dF(root1)
    tau2 = -0.5 / dF(root2)


    def test_methods(root_approx, second_point, third_point, tau, label, f_out):
        results = []
        try:
            x, it, err = simple_iteration(tau, root_approx, eps)
            results.append(("Проста ітерація", x, it, err))
        except Exception as e:
            results.append(("Проста ітерація", None, None, str(e)))
        try:
            x, it, err = newton(root_approx, eps)
            results.append(("Ньютон", x, it, err))
        except Exception as e:
            results.append(("Ньютон", None, None, str(e)))
        try:
            x, it, err = chebyshev(root_approx, eps)
            results.append(("Чебишев", x, it, err))
        except Exception as e:
            results.append(("Чебишев", None, None, str(e)))
        try:
            x, it, err = chord_method(root_approx, second_point, eps)
            results.append(("Хорд", x, it, err))
        except Exception as e:
            results.append(("Хорд", None, None, str(e)))
        try:
            x, it, err = parabola_method(root_approx, second_point, third_point, eps)
            results.append(("Парабол", x, it, err))
        except Exception as e:
            results.append(("Парабол", None, None, str(e)))
        try:
            x, it, err = inverse_interpolation_3point(root_approx, second_point, third_point, eps)
            results.append(("Звор. інтерполяція (3 т.)", x, it, err))
        except Exception as e:
            results.append(("Звор. інтерполяція (3 т.)", None, None, str(e)))
        f_out.write(f"\n--- {label} ---\n")
        for name, x, it, err in results:
            if isinstance(err, str):
                f_out.write(f"{name}: Помилка - {err}\n")
            else:
                f_out.write(f"{name}: корінь = {x:.12f}, ітерацій = {it}, похибка = {err:.2e}\n")


    with open("results_nonlinear.txt", "w") as fout:
        fout.write("Результати розв'язання рівняння F(x) = e^x - 3x = 0\n")
        fout.write("=" * 60 + "\n")
        x1_left = max(a, root1 - 0.05)
        x1_right = min(b, root1 + 0.05)
        test_methods(root1, x1_right, x1_left, tau1, "Корінь 1 (спадна функція)", fout)
        x2_left = max(a, root2 - 0.05)
        x2_right = min(b, root2 + 0.05)
        test_methods(root2, x2_right, x2_left, tau2, "Корінь 2 (зростаюча функція)", fout)

    with open("algebraic_coeffs.txt", "w") as f:
        f.write("1 0 1 0\n")

    coeffs = read_poly_coeffs("algebraic_coeffs.txt")
    x0_real = 0.5
    try:
        real_root, it_real, err_real = newton_horner(coeffs, x0_real, eps)
    except Exception as e:
        real_root, it_real, err_real = None, None, str(e)

    alpha0, beta0 = 0.0, 1.0
    try:
        alpha, beta, it_lin = lin_method(coeffs, alpha0, beta0, eps)
    except Exception as e:
        alpha, beta, it_lin = None, None, str(e)

    with open("results_algebraic.txt", "w") as f:
        f.write("Результати для алгебраїчного рівняння x^3 + x = 0\n")
        f.write("=" * 40 + "\n")
        if real_root:
            f.write(f"Дійсний корінь (Ньютон-Горнер): {real_root:.12f}, ітерацій: {it_real}, похибка: {err_real:.2e}\n")
        else:
            f.write("Дійсний корінь не знайдено.\n")
        if alpha:
            f.write(f"Комплексні корені (Лін): {alpha:.12f} ± {beta:.12f} i, ітерацій: {it_lin}\n")
        else:
            f.write("Комплексні корені не знайдено.\n")