import numpy as np

def simplex_method(c, A, b):
    m, n = A.shape
    J_B = list(range(n - m, n))      
    J_N = [j for j in range(n) if j not in J_B]
    
    B = A[:, J_B]
    B_inv = np.linalg.inv(B)
    x = np.zeros(n)

    while True:
        c_B = c[J_B]
        u = np.dot(c_B, B_inv)

        delta = { j: c[j] - np.dot(u, A[:, j]) for j in J_N }
        
        if all(r <= 0 for r in delta.values()):
            x[J_B] = np.dot(B_inv, b)
            return x
        
        j0_candidates = { j: r for j, r in delta.items() if r > 0 }
        j0 = max(j0_candidates, key=j0_candidates.get)

        y = np.dot(B_inv, A[:, j0])

        if all(y_i <= 0 for y_i in y):
            raise Exception("Unbounded solution")

        x_B = np.dot(B_inv, b)
        theta = [(x_B[i] / y[i]) if y[i] > 0 else np.inf for i in range(m)]
        theta_0 = min(theta)
        i0 = theta.index(theta_0)

        j_leave = J_B[i0]
        J_B[i0] = j0
        J_N[J_N.index(j0)] = j_leave

        B = A[:, J_B]
        B_inv = np.linalg.inv(B)

c = np.array([3, 2, 0, 0, 0], float)
A = np.array([
    [1, 1, 1, 0, 0],
    [2, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
], float)
b = np.array([4, 5, 2], float)

x_opt = simplex_method(c, A, b)
print("Optimal solution:", x_opt)
print("Optimal value:", np.dot(c, x_opt))