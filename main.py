import matplotlib.pyplot as plt
import problem
from solvers import RungeKutta4Solver, Adams2Solver


def plot_all(xv, exv, esv, hv, name):
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].plot(xv, exv, 'r');
    ax[0].set_title(f'{name}: Exact Error')
    ax[1].plot(xv, esv, 'b');
    ax[1].set_title(f'{name}: Estimated Error')
    ax[2].step(xv, hv, 'g');
    ax[2].set_title(f'{name}: Step Size h(x)')
    for a in ax: a.grid(True)
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    eps, h0 = 1e-4, 0.01

    rk = RungeKutta4Solver(problem.f, problem.y_exact)
    # Виклик solve_fixed для виконання п. 6-8 лаби
    rk.solve_fixed(problem.a, problem.b, problem.y0, h0)
    xr, yr, hr, esr, exr = rk.solve_auto(problem.a, problem.b, problem.y0, h0, eps)
    plot_all(xr, exr, esr, hr, "Runge-Kutta 4")

    ad = Adams2Solver(problem.f, problem.y_exact)
    xa, ya, ha, esa, exa = ad.solve_auto(problem.a, problem.b, problem.y0, h0, eps)
    plot_all(xa, exa, esa, ha, "Adams 2")