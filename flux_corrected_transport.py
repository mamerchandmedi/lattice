# %% Import package for numerical computations.
import numpy as np
# end: Import package for numerical computations.

# %% Import rescaled quantities.
from lattice import N, r
# end: Import rescaled quantities.

# %% Import rescaled quantities.
from scaled import tauZphi2Tpv_mod
# end: Import rescaled quantities.

# %% Importing characteristic variables.
from characteristic_variables import T_matrices2
# end: Importing characteristic variables.


def antidiffusive_matrix(KH, KL, FH, FL, dt):
    A = np.ndarray((N + 1, 3))
    A[0][-1] = 0
    A[0][0] = 0
    A[0][1] = dt*(KH[0][1]*(FH[1] - FH[0]) - KL[0][1]*(FL[1] - FL[0]))
    for j in range(1, N):
        A[j][-1] = dt*(KH[j][-1]*(FH[j-1] - FH[j]) - KL[j][-1]*(FL[j-1] - FL[j]))
        A[j][0] = 0
        A[j][1] = dt*(KH[j][1]*(FH[j+1] - FH[j]) - KL[j][1]*(FL[j+1] - FL[j]))
    A[N][-1] = dt*(KH[N][-1]*(FH[N-1]- FH[N]) - KL[N][-1]*(FL[N-1] - FL[N]))
    A[N][0] = 0
    A[N][1] = 0
    return A


def trans_antidiffusive_matrix(A1, A2, tau, Z, phi, T, P, v):
    AT1 = np.ndarray((N + 1, 3))
    AT2 = np.ndarray((N + 1, 3))
    Ti, Pi, vi = tauZphi2Tpv_mod(tau[0][0], Z[0][0], phi[0], T[0], P[0], v[0])
    T_mat, T_inv = T_matrices2(vi, phi[0], Ti)
    AT1[0][0] = T_inv[0][0] * A1[0][0] + T_inv[0][1] * A2[0][0]
    AT2[0][0] = T_inv[1][0] * A1[0][0] + T_inv[1][1] * A2[0][0]
    Ti, Pi, vi = tauZphi2Tpv_mod(tau[0][1], Z[0][1], phi[0], T[0], P[0], v[0])
    T_mat, T_inv = T_matrices2(vi, phi[0], Ti)
    AT1[0][1] = T_inv[0][0] * A1[0][1] + T_inv[0][1] * A2[0][1]
    AT2[0][1] = T_inv[1][0] * A1[0][1] + T_inv[1][1] * A2[0][1]
    for i in range(1, N):
        for j in [-1, 0, 1]:
            Ti, Pi, vi = \
                tauZphi2Tpv_mod(tau[i][j], Z[i][j], phi[i], T[i], P[i], v[i])
            T_mat, T_inv = T_matrices2(vi, phi[i], Ti)
            AT1[i][j] = T_inv[0][0] * A1[i][j] + T_inv[0][1] * A2[i][j]
            AT2[i][j] = T_inv[1][0] * A1[i][j] + T_inv[1][1] * A2[i][j]
    Ti, Pi, vi = \
        tauZphi2Tpv_mod(tau[N][-1], Z[N][-1], phi[N], T[N], P[N], v[N])
    T_mat, T_inv = T_matrices2(vi, phi[N], Ti)
    AT1[N][-1] = T_inv[0][0] * A1[N][-1] + T_inv[0][1] * A2[N][-1]
    AT2[N][-1] = T_inv[1][0] * A1[N][-1] + T_inv[1][1] * A2[N][-1]
    Ti, Pi, vi = tauZphi2Tpv_mod(tau[N][0], Z[N][0], phi[N], T[N], P[N], v[N])
    T_mat, T_inv = T_matrices2(vi, phi[N], Ti)
    AT1[N][0] = T_inv[0][0] * A1[N][0] + T_inv[0][1] * A2[N][0]
    AT2[N][0] = T_inv[1][0] * A1[N][0] + T_inv[1][1] * A2[N][0]
    return AT1, AT2


def P_vectors(A):
    P_plus = np.ndarray((N + 1,))
    P_minus = np.ndarray((N + 1,))
    P_plus[0] = max(0, A[0][1])
    P_minus[0] = min(0, A[0][1])
    for j in range(1, N):
        P_plus[j] = max(0, A[j][-1], A[j][1])
        P_minus[j] = min(0, A[j][-1], A[j][1])
    P_plus[N] = max(0, A[N][-1])
    P_minus[N] = min(0, A[N][-1])
    return P_plus, P_minus


def Q_values(U, i):
    if i == 0:
        Q_plus = max(U[i], U[i+1]) - U[i]
        Q_minus = min(U[i], U[i+1]) - U[i]
        return Q_plus, Q_minus
    elif i == N:
        Q_plus = max(U[i-1]*r[i-1]**2, U[i]*r[i]**2) - U[i]*r[i]**2
        Q_minus = min(U[i-1]*r[i-1]**2, U[i]*r[i]**2) - U[i]*r[i]**2
    else:
        Q_plus = max(U[i-1]*r[i-1]**2, U[i]*r[i]**2, U[i+1]*r[i+1]**2)\
                   - U[i]*r[i]**2
        Q_minus = min(U[i-1]*r[i-1]**2, U[i]*r[i]**2, U[i+1]*r[i+1]**2)\
            - U[i]*r[i]**2
    return Q_plus / r[i]**2, Q_minus / r[i]**2


def R_values(Qp, Qm, Pp, Pm):
    if Pp != 0:
        Rp = min(1, Qp/Pp)
    else:
        Rp = 1
    if Pm != 0:
        Rm = min(1, Qm/Pm)
    else:
        Rm = 1
    return Rp, Rm


def Ubar(U):
    U_matrix = np.ndarray((N + 1, 3))
    U_matrix[0][0] = U[0]
    U_matrix[0][1] = 0.5 * (U[0] + U[1])
    for j in range(1, N):
        U_matrix[j][-1] = 0.5 * (U[j] + U[j-1])
        U_matrix[j][0] = U[j]
        U_matrix[j][1] = 0.5 * (U[j] + U[j+1])
    U_matrix[N][-1] = 0.5 * (U[N] + U[N-1])
    U_matrix[N][0] = U[N]
    return U_matrix
