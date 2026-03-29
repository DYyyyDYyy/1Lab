import numpy as np
import matplotlib.pyplot as plt
import math

# Функція вологості
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

# --- Візуалізація вхідних даних ---
t_values = np.linspace(0, 20, 500) # Інтервал від 0 до 20
M_values = [M(t) for t in t_values]

fig_input, ax_input = plt.subplots(figsize=(8, 5))
ax_input.plot(t_values, M_values, color='#457b9d')
ax_input.set_title('Soil Moisture Model M(t)')
ax_input.set_xlabel('t')
ax_input.set_ylabel('M(t)')
ax_input.set_xlim(-1, 21)
ax_input.set_ylim(0, 52)
ax_input.grid(True, which='both', linestyle='-', linewidth=0.5)


t0 = 1.0
ax_input.plot(t0, M(t0), 'ro', label=f'Точка дослідження $t_0={t0}$')
ax_input.legend()

plt.show()

# --- 2. Точна (аналітична) похідна ---
def dM_exact(t):
    """M'(t) = -5 * e^(-0.1*t) + 5 * cos(t)"""
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


# --- 3. Чисельне диференціювання (центральна різниця) ---
def central_diff(func, x, h):
    """y_0'(h) = (f(x_0 + h) - f(x_0 - h)) / (2h)"""
    return (func(x + h) - func(x - h)) / (2 * h)


# --- 4. Головна функція ---
def main():
    x0 = 1.0  # Точка обчислення
    exact_val = dM_exact(x0)

    print(f"Точне значення похідної в точці x0={x0}: y'(x0) = {exact_val:.10f}\n")

    # =========================================================================
    # ПУНКТЫ 2 & 3: Дослідження залежності похибки від кроку h та пошук h0
    # =========================================================================
    # Діапазон h від 10^3 до 10^-20
    h_values = np.logspace(3, -20, num=100)
    errors_total = []

    best_h = None
    min_error = float('inf')
    best_diff_val = None

    for h in h_values:
        try:
            diff_val = central_diff(M, x0, h)
            error = abs(diff_val - exact_val)
            errors_total.append(error)

            if error < min_error:
                min_error = error
                best_h = h
                best_diff_val = diff_val
        except ZeroDivisionError:
            errors_total.append(np.nan)  # Ігноруємо ділення на нуль

    print(f"--- Результати пошуку оптимального кроку ---")
    print(f"Оптимальний крок h0 (експериментально): {best_h:e}")
    print(f"Мінімальна досягнута похибка R0: {min_error:e}")
    print(f"Значення похідної при h0: {best_diff_val:.10f}\n")

    # =========================================================================
    # ПУНКТ 6: Уточнення методом Рунге-Ромберга
    # =========================================================================
    h_fixed = 1e-3  # Крок h
    print(f"--- Уточнення методами для кроку h = {h_fixed} ---")

    y_prime_h = central_diff(M, x0, h_fixed)
    y_prime_2h = central_diff(M, x0, 2 * h_fixed)

    # Похибка при кроці h (R1)
    R1 = abs(y_prime_h - exact_val)

    # Уточнене значення (Рунге-Ромберг)
    y_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
    R2 = abs(y_R - exact_val)  #

    print(f"Похибка при h (R1): {R1:e}")
    print(f"Похибка Runge-Romberg (R2): {R2:e}")
    print(f"Ефект: похибка зменшилась у {R1 / R2:.1f} разів.\n")

    # =========================================================================
    # ПУНКТ 7: Метод Ейткена та оцінка порядку точності p
    # =========================================================================
    y_prime_4h = central_diff(M, x0, 4 * h_fixed)

    # Похибка Ейткена
    numerator = (y_prime_2h ** 2) - (y_prime_4h * y_prime_h)
    denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)

    y_E = numerator / denominator
    R3 = abs(y_E - exact_val)

    # Оцінка порядку точності p
    ratio = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
    p_aitken = math.log(ratio) / math.log(2)

    print(f"Похибка Aitken (R3): {R3:e}")
    print(f"Розрахунковий порядок точності p (Ейткен): {p_aitken:.5f}\n")

    # =========================================================================
    # ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ( matplotlib)
    # =========================================================================

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Аналіз методів чисельного диференціювання для M(t)\n(Точка x0={x0})', fontsize=16)

    # --- Графік 1: Залежність похибки від кроку h (log-log) ---
    axs[0, 0].loglog(h_values, errors_total, color='#1f77b4', label='Повна похибка')
    axs[0, 0].axvline(best_h, color='green', linestyle=':', label=f'Оптимальний $h_0$ ({best_h:.1e})')
    axs[0, 0].axhline(min_error, color='red', linestyle='--', label=f'Мін. похибка $R_0$ ({min_error:.1e})')
    axs[0, 0].set_xlabel('Крок $h$ (log scale)')
    axs[0, 0].set_ylabel('Абсолютна похибка $|y\'_{точно} - y\'_{чисел}|$')
    axs[0, 0].set_title('1. Error vs. Step Size (h)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.5)

    # --- Графік 2: Ефект Runge-Romberg ---
    methods_rr = ['Крок $h$', 'Рунге-Ромберг ($h, 2h$)']
    errors_rr = [R1, R2]
    axs[0, 1].bar(methods_rr, errors_rr, color=['#ff7f0e', '#2ca02c'])
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylabel('Абсолютна похибка (log scale)')
    axs[0, 1].set_title('2. Runge-Romberg Correction')
    axs[0, 1].set_ylim(min(errors_rr) * 0.1, max(errors_rr) * 10)
    for i, v in enumerate(errors_rr):
        axs[0, 1].text(i, v * 1.5, f'{v:.2e}', ha='center', va='bottom', fontsize=10)

    # --- Графік 3: Ефект Aitken ---
    methods_a = ['Крок $h$', 'Ейткен ($h, 2h, 4h$)']
    errors_a = [R1, R3]
    axs[1, 0].bar(methods_a, errors_a, color=['#ff7f0e', '#9467bd'])
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel('Абсолютна похибка (log scale)')
    axs[1, 0].set_title('3. Aitken Extrapolation')
    axs[1, 0].set_ylim(min(errors_a) * 0.1, max(errors_a) * 10)
    for i, v in enumerate(errors_a):
        axs[1, 0].text(i, v * 1.5, f'{v:.2e}', ha='center', va='bottom', fontsize=10)

    # --- Графік 4: Порядок точності ---
    theory_p = 2.0
    axs[1, 1].bar(['Теоретичний ($p$)', 'Розрахований Ейткеном ($\hat{p}$)'], [theory_p, p_aitken],
                  color=['grey', '#d62728'])
    axs[1, 1].set_ylim(0, 2.5)
    axs[1, 1].set_ylabel('Порядок точності $p$')
    axs[1, 1].set_title('4. Order of Accuracy p (Aitken)')
    for i, v in enumerate([theory_p, p_aitken]):
        axs[1, 1].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()