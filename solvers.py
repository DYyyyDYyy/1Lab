import problem

class ODESolver:
    def __init__(self, f, exact_f, name):
        self.f = f
        self.exact_f = exact_f
        self.name = name
        self.log_file = None

    def _open_log(self, filename):
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.log_file.write("Iter\tx\ty\th\tErr_Est\tExact_Err\n")

    def _log_step(self, i, x, y, h, err_est):
        exact_err = abs(y - self.exact_f(x))
        self.log_file.write(f"{i}\t{x:.6f}\t{y:.6f}\t{h:.6f}\t{err_est:.2e}\t{exact_err:.2e}\n")

    def _close_log(self):
        if self.log_file: self.log_file.close()

class RungeKutta4Solver(ODESolver):
    def __init__(self, f, exact_f):
        super().__init__(f, exact_f, "RK4")

    def step(self, x, y, h):
        k1 = self.f(x, y)
        k2 = self.f(x + h/2, y + h*k1/2)
        k3 = self.f(x + h/2, y + h*k2/2)
        k4 = self.f(x + h, y + h*k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def solve_fixed(self, a, b, y0, h):
        self._open_log(f"results_{self.name}_fixed.txt")
        x_v, y_v, ex_v = [a], [y0], [0.0]
        x, y = a, y0
        self._log_step(0, x, y, h, 0.0)
        i = 1
        while x < b:
            if x + h > b: h = b - x
            y = self.step(x, y, h)
            x += h
            x_v.append(x); y_v.append(y); ex_v.append(abs(y - self.exact_f(x)))
            self._log_step(i, x, y, h, 0.0)
            i += 1
        self._close_log()
        return x_v, y_v, ex_v

    def solve_auto(self, a, b, y0, h0, eps):
        self._open_log(f"results_{self.name}_auto.txt")
        x_v, y_v, h_v, es_v, ex_v = [a], [y0], [h0], [0.0], [0.0]
        x, y, h = a, y0, h0
        self._log_step(0, x, y, h, 0.0)
        i = 1
        while x < b:
            if x + h > b: h = b - x
            y_h = self.step(x, y, h)
            y_half = self.step(x + h/2, self.step(x, y, h/2), h/2)
            error_est = (16/15) * abs(y_h - y_half)
            if error_est > eps:
                h /= 2.0
                continue
            x += h
            y = y_h
            x_v.append(x); y_v.append(y); h_v.append(h)
            es_v.append(error_est); ex_v.append(abs(y - self.exact_f(x)))
            self._log_step(i, x, y, h, error_est)
            if error_est <= eps/32.0: h *= 2.0
            i += 1
        self._close_log()
        return x_v, y_v, h_v, es_v, ex_v

class Adams2Solver(ODESolver):
    def __init__(self, f, exact_f):
        super().__init__(f, exact_f, "Adams2")

    def step(self, xn, yn, xp, yp, h, eps_it=1e-7):
        fn, fp = self.f(xn, yn), self.f(xp, yp)
        ypred = yn + (h/2) * (3*fn - fp)
        xc = xn + h
        yc_p = ypred
        while True:
            yc = yn + (h/2) * (self.f(xc, yc_p) + fn)
            if abs(yc - yc_p) <= eps_it: break
            yc_p = yc
        return ypred, yc

    def solve_auto(self, a, b, y0, h0, eps):
        self._open_log(f"results_{self.name}_auto.txt")
        rk = RungeKutta4Solver(self.f, self.exact_f)
        x_v, y_v, h_v, es_v, ex_v = [a], [y0], [h0], [0.0], [0.0]
        x_p, y_p, h, y = a, y0, h0, rk.step(a, y0, h0)
        x = x_p + h
        x_v.append(x); y_v.append(y); h_v.append(h)
        ex_v.append(abs(y - self.exact_f(x))); es_v.append(0.0)
        self._log_step(0, x_p, y_p, h, 0.0); self._log_step(1, x, y, h, 0.0)
        i = 2
        while x < b:
            if x + h > b: h = b - x
            yp, yc = self.step(x, y, x_p, y_p, h)
            err = (1/6) * abs(yc - yp)
            if err > eps:
                h /= 2.0
                y = rk.step(x_p, y_p, h)
                x = x_p + h
                continue
            x_p, y_p, x, y = x, y, x + h, yc
            x_v.append(x); y_v.append(y); h_v.append(h)
            es_v.append(err); ex_v.append(abs(y - self.exact_f(x)))
            self._log_step(i, x, y, h, err)
            if err <= eps/8.0:
                h *= 2.0
                y = rk.step(x_p, y_p, h); x_p, y_p = x, y; x = x_p + h
            i += 1
        self._close_log()
        return x_v, y_v, h_v, es_v, ex_v