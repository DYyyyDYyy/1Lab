import requests
import numpy as np
import matplotlib.pyplot as plt



# ОТРИМАННЯ ДАНИХ (API)

def get_elevation_data():
    url = "https://api.open-elevation.com/api/v1/lookup"
    payload = {
        "locations": [
            {"latitude": 48.164214, "longitude": 24.536044},
            {"latitude": 48.164983, "longitude": 24.534836},
            {"latitude": 48.165605, "longitude": 24.534068},
            {"latitude": 48.166228, "longitude": 24.532915},
            {"latitude": 48.166777, "longitude": 24.531927},
            {"latitude": 48.167326, "longitude": 24.530884},
            {"latitude": 48.167011, "longitude": 24.530061},
            {"latitude": 48.166053, "longitude": 24.528039},
            {"latitude": 48.166655, "longitude": 24.526064},
            {"latitude": 48.166497, "longitude": 24.523574},
            {"latitude": 48.166128, "longitude": 24.520214},
            {"latitude": 48.165416, "longitude": 24.517170},
            {"latitude": 48.164546, "longitude": 24.514640},
            {"latitude": 48.163412, "longitude": 24.512980},
            {"latitude": 48.162331, "longitude": 24.511715},
            {"latitude": 48.162015, "longitude": 24.509462},
            {"latitude": 48.162147, "longitude": 24.506932},
            {"latitude": 48.161751, "longitude": 24.504244},
            {"latitude": 48.161197, "longitude": 24.501793},
            {"latitude": 48.160580, "longitude": 24.500537},
            {"latitude": 48.160250, "longitude": 24.500106}
        ]
    }
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["results"]
    except Exception as e:
        print(f"Помилка отримання даних: {e}")

        return [{"latitude": 0, "longitude": 0, "elevation": 0}] * 21


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prepare_data(results):
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = np.array([p["elevation"] for p in results])


    distances = [0]
    for i in range(1, len(coords)):
        d = haversine(*coords[i - 1], *coords[i])
        distances.append(distances[-1] + d)

    return np.array(distances), elevations



class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.h = np.diff(self.x)

        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.build_spline()

    def build_spline(self):
        n = self.n
        h = self.h
        y = self.y




        alpha = np.zeros(n)  # нижня діагональ
        beta = np.zeros(n)  # головна діагональ
        gamma = np.zeros(n)  # верхня діагональ
        delta = np.zeros(n)  # права частина (RHS)


        beta[0] = 1.0
        delta[0] = 0.0
        beta[n - 1] = 1.0
        delta[n - 1] = 0.0


        for i in range(1, n - 1):
            alpha[i] = h[i - 1]
            beta[i] = 2 * (h[i - 1] + h[i])
            gamma[i] = h[i]
            rhs = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
            delta[i] = rhs


        self.matrix_coeffs = (alpha, beta, gamma, delta)

        # Метод прогонки
        self.c = self.thomas_algorithm(alpha, beta, gamma, delta)


        self.a = y[:-1]
        self.b = np.zeros(n - 1)
        self.d = np.zeros(n - 1)

        for i in range(n - 1):

            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * h[i])


            self.b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (self.c[i + 1] + 2 * self.c[i])


    def thomas_algorithm(self, alpha, beta, gamma, delta):
        n = len(delta)

        A = np.zeros(n)
        B = np.zeros(n)

        A[0] = -gamma[0] / beta[0] if beta[0] != 0 else 0
        B[0] = delta[0] / beta[0] if beta[0] != 0 else 0

        for i in range(1, n - 1):
            denom = beta[i] + alpha[i] * A[i - 1]
            A[i] = -gamma[i] / denom
            B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom


        x = np.zeros(n)
        x[n - 1] = (delta[n - 1] - alpha[n - 1] * B[n - 2]) / (beta[n - 1] + alpha[n - 1] * A[n - 2])

        for i in range(n - 2, -1, -1):
            x[i] = A[i] * x[i + 1] + B[i]

        return x

    def evaluate(self, x_eval):


        if x_eval < self.x[0] or x_eval > self.x[-1]:
            return None


        i = np.searchsorted(self.x, x_eval) - 1
        if i < 0: i = 0
        if i >= len(self.b): i = len(self.b) - 1

        dx = x_eval - self.x[i]


        return self.a[i] + self.b[i] * dx + self.c[i] * (dx ** 2) + self.d[i] * (dx ** 3)

    def print_coefficients(self):
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТИ РОЗРАХУНКУ КОЕФІЦІЄНТІВ СПЛАЙНІВ")
        print("=" * 60)
        print(f"{'i':<3} | {'a_i':<10} | {'b_i':<10} | {'c_i':<10} | {'d_i':<10}")
        print("-" * 60)
        for i in range(len(self.b)):
            print(f"{i:<3} | {self.a[i]:<10.4f} | {self.b[i]:<10.4f} | {self.c[i]:<10.4f} | {self.d[i]:<10.6f}")



#  ГОЛОВНА ПРОГРАМА


def main():

    results = get_elevation_data()


    print("Дані успішно отримано! Кількість точок:", len(results))


    dist_full, elev_full = prepare_data(results)


    print("\nТабуляція (відстань, висота):")
    print(f"{'№':<3} | {'Distance (m)':<12} | {'Elevation (m)':<10}")
    for i in range(len(dist_full)):
        print(f"{i:<3d} | {dist_full[i]:<12.2f} | {elev_full[i]:<10.2f}")


    spline_full = CubicSpline(dist_full, elev_full)
    spline_full.print_coefficients()




    plt.figure(figsize=(14, 10))


    x_smooth = np.linspace(dist_full[0], dist_full[-1], 500)

    # 1. Повний набір
    y_smooth_full = [spline_full.evaluate(x) for x in x_smooth]
    plt.subplot(2, 2, 1)
    plt.plot(dist_full, elev_full, 'ro', label='Вузли (21 шт)')
    plt.plot(x_smooth, y_smooth_full, 'b-', label='Сплайн (Всі точки)')
    plt.title("Інтерполяція: Всі 21 точка")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Висота (м)")
    plt.legend()
    plt.grid(True)

    # 2. Зменшена кількість вузлів
    indices_10 = list(range(0, len(dist_full), 2))

    if indices_10[-1] != len(dist_full) - 1:
        indices_10.append(len(dist_full) - 1)

    dist_10 = dist_full[indices_10]
    elev_10 = elev_full[indices_10]
    spline_10 = CubicSpline(dist_10, elev_10)
    y_smooth_10 = [spline_10.evaluate(x) for x in x_smooth]

    plt.subplot(2, 2, 2)
    plt.plot(dist_full, elev_full, 'g.', alpha=0.3, label='Реальний рельєф')  # Фон
    plt.plot(dist_10, elev_10, 'ro', label=f'Вузли ({len(dist_10)} шт)')
    plt.plot(x_smooth, y_smooth_10, 'b--', label='Сплайн (Розріджений)')
    plt.title(f"Інтерполяція: ~10 точок")
    plt.xlabel("Відстань (м)")
    plt.grid(True)
    plt.legend()


    error = np.abs(np.array(y_smooth_full) - np.array(y_smooth_10))

    plt.subplot(2, 2, 3)
    plt.plot(x_smooth, error, 'r-')
    plt.title("Похибка інтерполяції (Abs Error)")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Похибка (м)")
    plt.grid(True)


    gradients = []
    for x in x_smooth:

        idx = np.searchsorted(spline_full.x, x) - 1
        if idx < 0: idx = 0
        if idx >= len(spline_full.b): idx = len(spline_full.b) - 1

        dx = x - spline_full.x[idx]
        grad = spline_full.b[idx] + 2 * spline_full.c[idx] * dx + 3 * spline_full.d[idx] * (dx ** 2)
        gradients.append(grad * 100)  # у відсотках

    plt.subplot(2, 2, 4)
    plt.plot(x_smooth, gradients, 'purple')
    plt.title("Градієнт (крутизна) маршруту %")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Ухил (%)")
    plt.grid(True)
    plt.axhline(0, color='black', lw=1)


    plt.tight_layout()


    # ДОДАТКОВІ ЗАВДАННЯ
    print("\n" + "=" * 60)
    print("ДОДАТКОВІ ЗАВДАННЯ")
    print("=" * 60)

    # 1. Загальна довжина
    print(f"Загальна довжина маршруту: {dist_full[-1]:.2f} м")

    # Набір і спуск
    total_ascent = np.sum(np.maximum(np.diff(elev_full), 0))
    total_descent = np.sum(np.maximum(-np.diff(elev_full), 0))
    print(f"Сумарний набір висоти: {total_ascent:.2f} м")
    print(f"Сумарний спуск: {total_descent:.2f} м")

    # 2. Статистика градієнта
    grad_arr = np.array(gradients)
    print(f"Максимальний підйом: {np.max(grad_arr):.2f} %")
    print(f"Максимальний спуск: {np.min(grad_arr):.2f} %")
    print(f"Середній градієнт (abs): {np.mean(np.abs(grad_arr)):.2f} %")

    # 3. Механічна енергія
    mass = 80
    g = 9.81
    energy_j = mass * g * total_ascent
    print(f"Механічна робота на підйом: {energy_j / 1000:.2f} кДж")
    print(f"Енергія в ккал: {energy_j / 4184:.2f} ккал")



    plt.show()


if __name__ == "__main__":
    main()