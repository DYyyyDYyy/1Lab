import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def f1(X):
    return X[0]**2 + X[1]**2 - 5

def f2(X):
    return X[0] - X[1] - 1

def Phi(X):
    return f1(X)**2 + f2(X)**2

def explore(X_base, delta_X, Phi, q, eps1, reduce_step=True):
    X_curr = X_base.copy()
    for i in range(len(X_curr)):
        success = False
        while not success:
            f_base = Phi(X_curr)
            X_curr[i] += delta_X[i]
            if Phi(X_curr) < f_base:
                success = True
                continue
            X_curr[i] -= delta_X[i]
            X_curr[i] -= delta_X[i]
            if Phi(X_curr) < f_base:
                success = True
                continue
            X_curr[i] += delta_X[i]
            if reduce_step:
                delta_X[i] /= q
                if delta_X[i] < eps1:
                    success = True
            else:
                success = True
    return X_curr

def hooke_jeeves(Phi, X0, delta_X0, q=2.0, p=2.0, eps1=1e-5, eps2=1e-5):
    X0 = np.array(X0, dtype=float)
    delta_X = np.array(delta_X0, dtype=float)
    trajectory = [X0.copy()]
    while True:
        X1 = explore(X0, delta_X, Phi, q, eps1, reduce_step=True)
        if np.array_equal(X1, X0):
            break
        trajectory.append(X1.copy())
        if np.linalg.norm(delta_X) < eps1 and abs(Phi(X1) - Phi(X0)) < eps2:
            break
        while True:
            Xp2 = X1 + p * (X1 - X0)
            X2 = explore(Xp2, delta_X, Phi, q, eps1, reduce_step=False)
            if Phi(X2) < Phi(X1):
                trajectory.append(X2.copy())
                X0 = X1.copy()
                X1 = X2.copy()
            else:
                X0 = X1.copy()
                break
    return X1, trajectory

X0_initial = [3.0, 3.0]
delta_X_initial = [1.0, 1.0]
q_param = 2.0
p_param = 2.0
epsilon1 = 1e-5
epsilon2 = 1e-5

best_X, traj = hooke_jeeves(Phi, X0_initial, delta_X_initial, q_param, p_param, epsilon1, epsilon2)

filename = "lab9_results.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write("Trajectory coordinates:\n")
    for i, pt in enumerate(traj):
        f.write(f"{i}: {pt[0]:.6f}, {pt[1]:.6f}, {Phi(pt):.6e}\n")
    f.write(f"\nSteps: {len(traj)}\n")
    f.write(f"Result: {best_X[0]:.5f}, {best_X[1]:.5f}\n")

x_vals = np.linspace(-4, 5, 400)
y_vals = np.linspace(-4, 5, 400)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
Z1 = f1([X_mesh, Y_mesh])
Z2 = f2([X_mesh, Y_mesh])
Z_phi = Phi([X_mesh, Y_mesh])

plt.figure(figsize=(10, 8))
plt.contour(X_mesh, Y_mesh, Z1, levels=[0], colors='blue')
plt.contour(X_mesh, Y_mesh, Z2, levels=[0], colors='green')
plt.contour(X_mesh, Y_mesh, Z_phi, levels=np.logspace(-1, 3, 15), cmap='viridis', alpha=0.3)
traj_x = [pt[0] for pt in traj]
traj_y = [pt[1] for pt in traj]
plt.plot(traj_x, traj_y, 'ro-', markersize=4)
plt.plot(best_X[0], best_X[1], 'r*', markersize=12)
plt.grid(True)
plt.show()