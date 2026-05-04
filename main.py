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


def simple_iteration(tau, x0, eps=1e-10, max_iter=1000, verbose=True):
    x = x0
    if verbose:
        print(f"Початкове наближення: x0 = {x:.12f}")
    for i in range(max_iter):
        x_new = x + tau * F(x)
        if verbose:
            print(f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |x_new - x| = {abs(x_new - x):.12e}")
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод простої ітерації не збігся")


def newton(x0, eps=1e-10, max_iter=1000, verbose=True):
    x = x0
    if verbose:
        print(f"Початкове наближення: x0 = {x:.12f}")
    for i in range(max_iter):
        fx = F(x)
        dfx = dF(x)
        x_new = x - fx / dfx
        if verbose:
            print(f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |x_new - x| = {abs(x_new - x):.12e}")
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод Ньютона не збігся")


def chebyshev(x0, eps=1e-10, max_iter=1000, verbose=True):
    x = x0
    if verbose:
        print(f"Початкове наближення: x0 = {x:.12f}")
    for i in range(max_iter):
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        correction = fx / dfx + 0.5 * (fx ** 2) * d2fx / (dfx ** 3)
        x_new = x - correction
        if verbose:
            print(f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |x_new - x| = {abs(x_new - x):.12e}")
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(F(x_new))
        x = x_new
    raise RuntimeError("Метод Чебишева не збігся")


def chord_method(x0, x1, eps=1e-10, max_iter=1000, verbose=True):
    x_prev, x_curr = x0, x1
    if verbose:
        print(f"Початкові наближення: x0 = {x0:.12f}, x1 = {x1:.12f}")
    for i in range(max_iter):
        f_prev = F(x_prev)
        f_curr = F(x_curr)
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if verbose:
            print(
                f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |x_new - x_curr| = {abs(x_new - x_curr):.12e}")
        if abs(F(x_new)) < eps and abs(x_new - x_curr) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(F(x_new))
        x_prev, x_curr = x_curr, x_new
    raise RuntimeError("Метод хорд не збігся")


def parabola_method(x0, x1, x2, eps=1e-10, max_iter=1000, verbose=True):
    x = [x0, x1, x2]
    if verbose:
        print(f"Початкові наближення: x0 = {x0:.12f}, x1 = {x1:.12f}, x2 = {x2:.12f}")
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
        if verbose:
            print(f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |dx| = {abs(dx):.12e}")
        if abs(F(x_new)) < eps and abs(dx) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(F(x_new))
        x = [x[1], x[2], x_new]
    raise RuntimeError("Метод парабол не збігся")


def inverse_interpolation_3point(x0, x1, x2, eps=1e-10, max_iter=1000, verbose=True):
    x = [x0, x1, x2]
    if verbose:
        print(f"Початкові наближення: x0 = {x0:.12f}, x1 = {x1:.12f}, x2 = {x2:.12f}")
    for i in range(max_iter):
        y0, y1, y2 = F(x[0]), F(x[1]), F(x[2])
        denom1 = (y0 - y1) * (y0 - y2)
        denom2 = (y1 - y0) * (y1 - y2)
        denom3 = (y2 - y0) * (y2 - y1)
        x_new = (y1 * y2 * x[0]) / denom1 + (y0 * y2 * x[1]) / denom2 + (y0 * y1 * x[2]) / denom3
        if verbose:
            print(
                f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {F(x_new):.12e}, |x_new - x[2]| = {abs(x_new - x[2]):.12e}")
        if abs(F(x_new)) < eps and abs(x_new - x[2]) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
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


def newton_horner(coeffs, x0, eps=1e-10, max_iter=1000, verbose=True):
    x = x0
    if verbose:
        print(f"Початкове наближення: x0 = {x:.12f}")
    for i in range(max_iter):
        val, deriv = horner_value_and_derivative(coeffs, x)
        x_new = x - val / deriv
        if verbose:
            print(
                f"Ітерація {i + 1}: x = {x_new:.12f}, F(x) = {horner_eval(coeffs, x_new):.12e}, |x_new - x| = {abs(x_new - x):.12e}")
        if abs(horner_eval(coeffs, x_new)) < eps and abs(x_new - x) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {i + 1}")
            return x_new, i + 1, abs(horner_eval(coeffs, x_new))
        x = x_new
    raise RuntimeError("Метод Ньютона (Горнер) не збігся")


def lin_method(coeffs, alpha0, beta0, eps=1e-10, max_iter=1000, verbose=True):
    n = len(coeffs) - 1
    alpha, beta = alpha0, beta0
    p = -2 * alpha
    q = alpha ** 2 + beta ** 2
    if verbose:
        print(f"Початкові наближення: α0 = {alpha:.12f}, β0 = {beta:.12f}")
        print(f"Початкові p = {p:.12f}, q = {q:.12f}")
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
        if verbose:
            print(f"Ітерація {it + 1}: α = {alpha_new:.12f}, β = {beta_new:.12f}, p = {p_new:.12f}, q = {q_new:.12f}")
            print(f"  |α_new - α| = {abs(alpha_new - alpha):.12e}, |β_new - β| = {abs(beta_new - beta):.12e}")
        if abs(alpha_new - alpha) < eps and abs(beta_new - beta) < eps:
            if verbose:
                print(f"Збіжність досягнута на ітерації {it + 1}")
            return alpha_new, beta_new, it + 1
        alpha, beta = alpha_new, beta_new
        p, q = p_new, q_new
    raise RuntimeError("Метод Ліна не збігся")


if __name__ == "__main__":
    a, b = 0.0, 2.0
    h = 0.1
    eps = 1e-10

    print("=" * 80)
    print("ТАБУЛЯЦІЯ ФУНКЦІЇ F(x) = e^x - 3x")
    print("=" * 80)
    roots_approx = tabulate_and_find_roots(a, b, h)
    print(f"Знайдено наближених коренів: {len(roots_approx)}")
    for idx, root in enumerate(roots_approx):
        print(f"  Корінь {idx + 1}: x ≈ {root:.10f}")

    if len(roots_approx) < 2:
        raise SystemExit("Не знайдено двох коренів!")

    root1 = roots_approx[0]
    root2 = roots_approx[1]
    if dF(root1) > 0:
        root1, root2 = root2, root1

    print(f"\nКорінь 1 (спадна ділянка, F'(x) < 0): x ≈ {root1:.10f}")
    print(f"Корінь 2 (зростаюча ділянка, F'(x) > 0): x ≈ {root2:.10f}")

    tau1 = -0.5 / dF(root1)
    tau2 = -0.5 / dF(root2)

    print(f"\nПараметри простої ітерації:")
    print(f"  Для кореня 1: τ = {tau1:.6f}")
    print(f"  Для кореня 2: τ = {tau2:.6f}")


    def test_methods(root_approx, second_point, third_point, tau, label, f_out):
        f_out.write(f"\n{'=' * 60}\n")
        f_out.write(f"{label}\n")
        f_out.write(f"{'=' * 60}\n")

        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")

        methods = [
            ("МЕТОД ПРОСТОЇ ІТЕРАЦІЇ", lambda: simple_iteration(tau, root_approx, eps, verbose=False)),
            ("МЕТОД НЬЮТОНА", lambda: newton(root_approx, eps, verbose=False)),
            ("МЕТОД ЧЕБИШЕВА", lambda: chebyshev(root_approx, eps, verbose=False)),
            ("МЕТОД ХОРД", lambda: chord_method(root_approx, second_point, eps, verbose=False)),
            ("МЕТОД ПАРАБОЛ", lambda: parabola_method(root_approx, second_point, third_point, eps, verbose=False)),
            ("МЕТОД ЗВОРОТНОЇ ІНТЕРПОЛЯЦІЇ",
             lambda: inverse_interpolation_3point(root_approx, second_point, third_point, eps, verbose=False))
        ]

        for name, method_func in methods:
            print(f"\n--- {name} ---")
            f_out.write(f"\n{name}\n")

            original_verbose = True

            if name == "МЕТОД ПРОСТОЇ ІТЕРАЦІЇ":
                x, it, err = simple_iteration(tau, root_approx, eps, verbose=True)
            elif name == "МЕТОД НЬЮТОНА":
                x, it, err = newton(root_approx, eps, verbose=True)
            elif name == "МЕТОД ЧЕБИШЕВА":
                x, it, err = chebyshev(root_approx, eps, verbose=True)
            elif name == "МЕТОД ХОРД":
                x, it, err = chord_method(root_approx, second_point, eps, verbose=True)
            elif name == "МЕТОД ПАРАБОЛ":
                x, it, err = parabola_method(root_approx, second_point, third_point, eps, verbose=True)
            elif name == "МЕТОД ЗВОРОТНОЇ ІНТЕРПОЛЯЦІЇ":
                x, it, err = inverse_interpolation_3point(root_approx, second_point, third_point, eps, verbose=True)

            f_out.write(f"  Корінь: {x:.15f}\n")
            f_out.write(f"  Кількість ітерацій: {it}\n")
            f_out.write(f"  Похибка |F(x)|: {err:.15e}\n")
            print(f"  ОСТАТОЧНИЙ РЕЗУЛЬТАТ: x = {x:.15f}, ітерацій: {it}, похибка: {err:.15e}\n")


    with open("results_nonlinear.txt", "w", encoding="utf-8") as fout:
        fout.write("Результати розв'язання рівняння F(x) = e^x - 3x = 0\n")
        fout.write("=" * 60 + "\n")

        x1_left = max(a, root1 - 0.05)
        x1_right = min(b, root1 + 0.05)
        test_methods(root1, x1_right, x1_left, tau1, "КОРІНЬ 1 (спадна функція, F'(x) < 0)", fout)

        x2_left = max(a, root2 - 0.05)
        x2_right = min(b, root2 + 0.05)
        test_methods(root2, x2_right, x2_left, tau2, "КОРІНЬ 2 (зростаюча функція, F'(x) > 0)", fout)

    print("\n" + "=" * 80)
    print("АЛГЕБРАЇЧНЕ РІВНЯННЯ ТРЕТЬОГО СТЕПЕНЯ")
    print("=" * 80)

    with open("algebraic_coeffs.txt", "w") as f:
        f.write("1 0 1 0\n")

    coeffs = read_poly_coeffs("algebraic_coeffs.txt")
    print(f"Коефіцієнти многочлена: {coeffs}")
    print(f"Рівняння: {coeffs[0]}x³ + {coeffs[1]}x² + {coeffs[2]}x + {coeffs[3]} = 0")
    print("Спрощено: x³ + x = 0")

    print("\n--- ЗНАХОДЖЕННЯ ДІЙСНОГО КОРЕНЯ (МЕТОД НЬЮТОНА-ГОРНЕРА) ---")
    x0_real = 0.5
    try:
        real_root, it_real, err_real = newton_horner(coeffs, x0_real, eps, verbose=True)
        print(f"\nОСТАТОЧНИЙ РЕЗУЛЬТАТ: x = {real_root:.15f}, ітерацій: {it_real}, похибка: {err_real:.15e}")
    except Exception as e:
        print(f"Помилка: {e}")
        real_root, it_real, err_real = None, None, str(e)

    print("\n--- ЗНАХОДЖЕННЯ КОМПЛЕКСНИХ КОРЕНІВ (МЕТОД ЛІНА) ---")
    alpha0, beta0 = 0.0, 1.0
    try:
        alpha, beta, it_lin = lin_method(coeffs, alpha0, beta0, eps, verbose=True)
        print(f"\nОСТАТОЧНИЙ РЕЗУЛЬТАТ: x = {alpha:.15f} ± {beta:.15f} i, ітерацій: {it_lin}")
    except Exception as e:
        print(f"Помилка: {e}")
        alpha, beta, it_lin = None, None, str(e)

    with open("results_algebraic.txt", "w", encoding="utf-8") as f:
        f.write("Результати для алгебраїчного рівняння x³ + x = 0\n")
        f.write("=" * 40 + "\n")
        if real_root:
            f.write(f"Дійсний корінь (Ньютон-Горнер): {real_root:.15f}\n")
            f.write(f"Кількість ітерацій: {it_real}\n")
            f.write(f"Похибка: {err_real:.15e}\n")
        else:
            f.write("Дійсний корінь не знайдено.\n")
        if alpha:
            f.write(f"\nКомплексні корені (Лін): {alpha:.15f} ± {beta:.15f} i\n")
            f.write(f"Кількість ітерацій: {it_lin}\n")
        else:
            f.write("\nКомплексні корені не знайдено.\n")

    print("\n" + "=" * 80)
    print("ВИКОНАННЯ ЗАВЕРШЕНО")
    print("=" * 80)
    print("Результати збережено у файлах:")
    print("  - tabulation.txt (табуляція функції)")
    print("  - results_nonlinear.txt (результати всіх методів)")
    print("  - results_algebraic.txt (результати алгебраїчного рівняння)")