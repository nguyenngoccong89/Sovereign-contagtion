# ============================================================
# Plot bond price function for O as a function of debt choices
# Blue line: P has market access (joint access)
# Orange line: P has no market access (sole O)
# ============================================================

import numpy as np
import math
from numba import njit, prange
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# Parameters
# ============================================================

# Household / govt
sigmaH = 2.0
theta  = 1.0
nu     = 2.0
omega_g = 0.40
beta   = 0.90

# Shocks
rho_z  = 0.90
sigma_eps = 0.02
NZ = 5  # Use smaller grid for faster computation
tauchen_m = 3.0

# Default cap rule
delta = 0.90

# Market access
lam = 0.20

# Lender
sigmaL = 2.5
betaL  = 0.98
yL     = 0.80

# Grids
NB = 7
NT = 7
bmax = 0.8
taumax = 0.8

# EV + damping
rhoA   = 0.02
rhotau = 0.02
rhoD   = 0.02
xi_q   = 0.30  # damping for MU
xi_V   = 0.30  # damping for value/default

EULER_GAMMA = 0.5772156649015329

# ============================================================
# Tauchen discretization
# ============================================================

def tauchen(N, rho, sigma, m=3.0):
    std_y = sigma / math.sqrt(1 - rho**2)
    y_max = m * std_y
    y_min = -y_max
    y_grid = np.linspace(y_min, y_max, N)
    step = (y_max - y_min) / (N - 1)

    P = np.zeros((N, N))
    from math import erf, sqrt

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    for i in range(N):
        for j in range(N):
            if j == 0:
                z = (y_grid[0] - rho * y_grid[i] + step/2) / sigma
                P[i, j] = norm_cdf(z)
            elif j == N-1:
                z = (y_grid[N-1] - rho * y_grid[i] - step/2) / sigma
                P[i, j] = 1.0 - norm_cdf(z)
            else:
                z_up = (y_grid[j] - rho * y_grid[i] + step/2) / sigma
                z_lo = (y_grid[j] - rho * y_grid[i] - step/2) / sigma
                P[i, j] = norm_cdf(z_up) - norm_cdf(z_lo)

    z_grid = np.exp(y_grid)
    return z_grid, P

def stationary_dist(P, tol=1e-14, maxit=200000):
    n = P.shape[0]
    pi = np.ones(n)/n
    for _ in range(maxit):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi

z_grid, Pz = tauchen(NZ, rho_z, sigma_eps, tauchen_m)
pi_z = stationary_dist(Pz)
zbar = float(np.sum(pi_z * z_grid))
z_def = np.minimum(z_grid, delta * zbar)

print("z_grid:", z_grid, flush=True)

# ============================================================
# Joint transition and grids
# ============================================================

Pzz = np.kron(Pz, Pz)
NZ2 = NZ*NZ

izO_of_zz = np.empty(NZ2, dtype=np.int64)
izP_of_zz = np.empty(NZ2, dtype=np.int64)
k = 0
for izP in range(NZ):
    for izO in range(NZ):
        izO_of_zz[k] = izO
        izP_of_zz[k] = izP
        k += 1

b_grid = np.linspace(0.0, bmax, NB)
tau_grid = np.linspace(0.0, taumax, NT)

NA = NB * NT
bprime_of_a = np.empty(NA, dtype=np.int64)
tau_of_a    = np.empty(NA, dtype=np.int64)

k = 0
for ibp in range(NB):
    for it in range(NT):
        bprime_of_a[k] = ibp
        tau_of_a[k] = it
        k += 1

print("NB, NT, NA:", NB, NT, NA, flush=True)

# ============================================================
# Household block
# ============================================================

@njit
def v_g(g):
    if g <= 0.0:
        return -1e18
    return omega_g * (g**(1.0 - sigmaH)) / (1.0 - sigmaH)

@njit
def u_ghh(c, l):
    X = c - theta * (l**(1.0 + nu)) / (1.0 + nu)
    if X <= 0.0:
        return -1e18, X
    return (X**(1.0 - sigmaH)) / (1.0 - sigmaH), X

@njit
def household_block(z, tau):
    l_int = (z / (theta * (1.0 + tau)))**(1.0/nu)
    l = l_int
    if l > 1.0:
        l = 1.0
    c = z * l / (1.0 + tau)
    u, X = u_ghh(c, l)
    rev = tau * c
    ok = 1
    if X <= 0.0:
        ok = 0
    return l, c, u, rev, ok

# Precompute household blocks
l_rep   = np.zeros((NZ, NT))
c_rep   = np.zeros((NZ, NT))
u_rep   = np.zeros((NZ, NT))
rev_rep = np.zeros((NZ, NT))
ok_rep  = np.zeros((NZ, NT), dtype=np.int8)

l_defA   = np.zeros((NZ, NT))
c_defA   = np.zeros((NZ, NT))
u_defA   = np.zeros((NZ, NT))
rev_defA = np.zeros((NZ, NT))
ok_defA  = np.zeros((NZ, NT), dtype=np.int8)

for iz in range(NZ):
    for it in range(NT):
        l,c,u,rev,ok = household_block(float(z_grid[iz]), float(tau_grid[it]))
        l_rep[iz,it]=l; c_rep[iz,it]=c; u_rep[iz,it]=u; rev_rep[iz,it]=rev; ok_rep[iz,it]=ok

        l,c,u,rev,ok = household_block(float(z_def[iz]), float(tau_grid[it]))
        l_defA[iz,it]=l; c_defA[iz,it]=c; u_defA[iz,it]=u; rev_defA[iz,it]=rev; ok_defA[iz,it]=ok

print("Household blocks precomputed.", flush=True)

# ============================================================
# State space
# ============================================================

state_zz = []
state_ibO = []
state_ibP = []
state_r = []

def add_states_for_regime(r, mO, mP):
    for izz in range(NZ2):
        ibO_list = range(NB) if mO==0 else [0]
        ibP_list = range(NB) if mP==0 else [0]
        for ibO in ibO_list:
            for ibP in ibP_list:
                state_zz.append(izz)
                state_ibO.append(ibO)
                state_ibP.append(ibP)
                state_r.append(r)

add_states_for_regime(1,0,0)
add_states_for_regime(2,0,1)
add_states_for_regime(3,1,0)
add_states_for_regime(4,1,1)

state_zz = np.array(state_zz, dtype=np.int64)
state_ibO = np.array(state_ibO, dtype=np.int64)
state_ibP = np.array(state_ibP, dtype=np.int64)
state_r  = np.array(state_r, dtype=np.int64)
NS = state_zz.size

sid_of = -np.ones((NZ2, NB, NB, 5), dtype=np.int64)
for sid in range(NS):
    sid_of[state_zz[sid], state_ibO[sid], state_ibP[sid], state_r[sid]] = sid

print("NS:", NS, flush=True)

# ============================================================
# EV operators and price solvers
# ============================================================

@njit
def ev_logsumexp(vals, rho):
    m = -1e30
    for i in range(vals.size):
        if vals[i] > m:
            m = vals[i]
    if m <= -1e20:
        return -1e18
    s = 0.0
    inv = 1.0 / rho
    for i in range(vals.size):
        s += math.exp((vals[i] - m) * inv)
    return EULER_GAMMA*rho + rho*(math.log(s) + m*inv)

@njit
def logit_prob(Vdef, Vrep, rhoD_):
    a = Vdef / rhoD_
    b = Vrep / rhoD_
    m = a if a > b else b
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    return ea / (ea + eb)

@njit
def inclusive_defrep(Vdef, Vrep, rhoD_):
    a = Vdef / rhoD_
    b = Vrep / rhoD_
    m = a if a > b else b
    return EULER_GAMMA*rhoD_ + rhoD_*(math.log(math.exp(a-m)+math.exp(b-m)) + m)

@njit
def decode_regime(r):
    if r == 1:
        return 0,0
    if r == 2:
        return 0,1
    if r == 3:
        return 1,0
    return 1,1

@njit
def next_regime_after_today(m_today, default_today):
    if m_today == 0:
        return 1 if default_today==1 else 0
    else:
        return 1

@njit
def sid_lookup(izz, ibO, ibP, r, sid_of_):
    return sid_of_[izz, ibO, ibP, r]

@njit
def exp_next_objects(izz, mO_today, mP_today, dO_today, dP_today,
                     ibOprime, ibPprime,
                     V_O, V_P, d_O, d_P, muL, sid_of_):
    EV_O = 0.0
    EV_P = 0.0
    EmupayO = 0.0
    EmupayP = 0.0

    mO_next_det = next_regime_after_today(mO_today, dO_today)
    mP_next_det = next_regime_after_today(mP_today, dP_today)

    for izz2 in range(NZ2):
        pz = Pzz[izz, izz2]
        if pz == 0.0:
            continue

        for reO in range(2):
            if mO_today == 1:
                pO = lam if reO==1 else (1.0-lam)
                mO_next = 0 if reO==1 else 1
                ibO_next = 0
            else:
                pO = 1.0 if reO==0 else 0.0
                mO_next = mO_next_det
                ibO_next = ibOprime if mO_next==0 else 0
            if pO == 0.0:
                continue

            for reP in range(2):
                if mP_today == 1:
                    pP = lam if reP==1 else (1.0-lam)
                    mP_next = 0 if reP==1 else 1
                    ibP_next = 0
                else:
                    pP = 1.0 if reP==0 else 0.0
                    mP_next = mP_next_det
                    ibP_next = ibPprime if mP_next==0 else 0
                if pP == 0.0:
                    continue

                prob = pz * pO * pP

                if mO_next==0 and mP_next==0:
                    r_next = 1
                elif mO_next==0 and mP_next==1:
                    r_next = 2
                elif mO_next==1 and mP_next==0:
                    r_next = 3
                else:
                    r_next = 4

                sid2 = sid_lookup(izz2, ibO_next, ibP_next, r_next, sid_of_)
                EV_O += prob * V_O[sid2]
                EV_P += prob * V_P[sid2]
                EmupayO += prob * muL[sid2] * (1.0 - d_O[sid2])
                EmupayP += prob * muL[sid2] * (1.0 - d_P[sid2])

    return EV_O, EV_P, EmupayO, EmupayP

@njit
def solve_price_single(b_i, bprime_i, N_i):
    if bprime_i == 0.0:
        CL = yL + b_i
        if CL <= 0.0:
            return 0.0
        return N_i * (CL**sigmaL)
    if N_i <= 1e-16:
        return 0.0

    q_lo = 0.0
    q_hi = 2.0
    for _ in range(60):
        CL = yL + b_i - q_hi*bprime_i
        if CL <= 1e-14:
            q_hi *= 0.5
            continue
        f_hi = q_hi * (CL**(-sigmaL)) - N_i
        if f_hi >= 0.0:
            break
        q_hi *= 2.0

    for _ in range(80):
        q_mid = 0.5*(q_lo+q_hi)
        CL = yL + b_i - q_mid*bprime_i
        if CL <= 1e-14:
            q_hi = q_mid
            continue
        f_mid = q_mid*(CL**(-sigmaL)) - N_i
        if f_mid > 0.0:
            q_hi = q_mid
        else:
            q_lo = q_mid
    return 0.5*(q_lo+q_hi)

@njit
def solve_price_RR(bO, bP, bOprime, bPprime, NO, NP):
    if NO <= 1e-16:
        return 0.0, 0.0
    ratio = NP/NO if NP>1e-16 else 0.0

    q_lo = 0.0
    q_hi = 2.0
    for _ in range(80):
        qP = ratio*q_hi
        CL = yL + bO + bP - q_hi*bOprime - qP*bPprime
        if CL <= 1e-14:
            q_hi *= 0.5
            continue
        f_hi = q_hi*(CL**(-sigmaL)) - NO
        if f_hi >= 0.0:
            break
        q_hi *= 2.0

    for _ in range(90):
        q_mid = 0.5*(q_lo+q_hi)
        qP = ratio*q_mid
        CL = yL + bO + bP - q_mid*bOprime - qP*bPprime
        if CL <= 1e-14:
            q_hi = q_mid
            continue
        f_mid = q_mid*(CL**(-sigmaL)) - NO
        if f_mid > 0.0:
            q_hi = q_mid
        else:
            q_lo = q_mid

    qO = 0.5*(q_lo+q_hi)
    return qO, ratio*qO

print("Compiled price solvers.", flush=True)

# ============================================================
# Fixed point iteration (same as main solver)
# ============================================================

V_O = np.zeros(NS)
V_P = np.zeros(NS)
d_O = np.zeros(NS)
d_P = np.zeros(NS)
muL = np.ones(NS) * (yL**(-sigmaL))

@njit(parallel=True)
def fixed_point_update(V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_):
    V_O_new = np.empty_like(V_O_old)
    V_P_new = np.empty_like(V_P_old)
    d_O_new = np.empty_like(d_O_old)
    d_P_new = np.empty_like(d_P_old)
    mu_tilde = np.empty_like(muL_old)

    for sid in prange(NS):
        izz = state_zz[sid]
        ibO = state_ibO[sid]
        ibP = state_ibP[sid]
        r   = state_r[sid]
        mO, mP = decode_regime(r)

        izO = izO_of_zz[izz]
        izP = izP_of_zz[izz]
        bO = b_grid[ibO]
        bP = b_grid[ibP]

        # r=4 both excluded
        if r == 4:
            valsO = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    valsO[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsO[it] = u + v_g(g) + beta*EV_O_cont
            VO = ev_logsumexp(valsO, rhotau)

            valsP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    valsP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsP[it] = u + v_g(g) + beta*EV_P_cont
            VP = ev_logsumexp(valsP, rhotau)

            V_O_new[sid] = VO
            V_P_new[sid] = VP
            d_O_new[sid] = 0.0
            d_P_new[sid] = 0.0
            CL = yL
            mu_tilde[sid] = CL**(-sigmaL)
            continue

        # r=2 sole O (P excluded)
        if r == 2:
            valsP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    valsP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _ = exp_next_objects(izz, 0, 1, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsP[it] = u + v_g(g) + beta*EV_P_cont
            V_P_new[sid] = ev_logsumexp(valsP, rhotau)
            d_P_new[sid] = 0.0

            Wrep = np.empty(NA)
            for aO in range(NA):
                ibOp = bprime_of_a[aO]
                itauO = tau_of_a[aO]
                if ok_rep[izO,itauO] == 0:
                    Wrep[aO] = -1e18
                    continue
                bOp = b_grid[ibOp]

                EV_O_cont, _, EmupayO, _ = exp_next_objects(
                    izz, 0, 1, 0, 0, ibOp, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NO = betaL*EmupayO
                qO = solve_price_single(bO, bOp, NO)

                gO = rev_rep[izO,itauO] + qO*bOp - bO
                if gO <= 0.0:
                    Wrep[aO] = -1e18
                    continue
                Wrep[aO] = u_rep[izO,itauO] + v_g(gO) + beta*EV_O_cont

            VrepO = ev_logsumexp(Wrep, rhoA)

            Wdef_tau = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    Wdef_tau[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _ = exp_next_objects(
                    izz, 0, 1, 1, 0, 0, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tau[it] = u + v_g(g) + beta*EV_O_cont

            VdefO = ev_logsumexp(Wdef_tau, rhotau)
            dO = logit_prob(VdefO, VrepO, rhoD)
            VO = inclusive_defrep(VdefO, VrepO, rhoD)

            V_O_new[sid] = VO
            d_O_new[sid] = dO

            mW = -1e30
            for aO in range(NA):
                if Wrep[aO] > mW:
                    mW = Wrep[aO]
            S = 0.0
            Eqb = 0.0
            if mW > -1e20:
                for aO in range(NA):
                    if Wrep[aO] <= -1e17:
                        continue
                    w = math.exp((Wrep[aO]-mW)/rhoA)
                    S += w
                    ibOp = bprime_of_a[aO]
                    bOp = b_grid[ibOp]
                    EV_O_cont, _, EmupayO, _ = exp_next_objects(
                        izz, 0, 1, 0, 0, ibOp, 0,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NO = betaL*EmupayO
                    qO = solve_price_single(bO, bOp, NO)
                    Eqb += w*(qO*bOp)
                Eqb = Eqb/S if S>0 else 0.0

            CL = yL + (1.0-dO)*bO - (1.0-dO)*Eqb
            if CL <= 0.01:
                CL = 0.01
            mu_tilde[sid] = CL**(-sigmaL)
            continue

        # r=3 sole P (O excluded)
        if r == 3:
            valsO = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    valsO[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _ = exp_next_objects(izz, 1, 0, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsO[it] = u + v_g(g) + beta*EV_O_cont
            V_O_new[sid] = ev_logsumexp(valsO, rhotau)
            d_O_new[sid] = 0.0

            Wrep = np.empty(NA)
            for aP in range(NA):
                ibPp = bprime_of_a[aP]
                itauP = tau_of_a[aP]
                if ok_rep[izP,itauP] == 0:
                    Wrep[aP] = -1e18
                    continue
                bPp = b_grid[ibPp]

                _, EV_P_cont, _, EmupayP = exp_next_objects(
                    izz, 1, 0, 0, 0, 0, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NP = betaL*EmupayP
                qP = solve_price_single(bP, bPp, NP)

                gP = rev_rep[izP,itauP] + qP*bPp - bP
                if gP <= 0.0:
                    Wrep[aP] = -1e18
                    continue
                Wrep[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont

            VrepP = ev_logsumexp(Wrep, rhoA)

            Wdef_tau = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    Wdef_tau[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _ = exp_next_objects(
                    izz, 1, 0, 0, 1, 0, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tau[it] = u + v_g(g) + beta*EV_P_cont

            VdefP = ev_logsumexp(Wdef_tau, rhotau)
            dP = logit_prob(VdefP, VrepP, rhoD)
            VP = inclusive_defrep(VdefP, VrepP, rhoD)

            V_P_new[sid] = VP
            d_P_new[sid] = dP

            mW = -1e30
            for aP in range(NA):
                if Wrep[aP] > mW:
                    mW = Wrep[aP]
            S = 0.0
            Eqb = 0.0
            if mW > -1e20:
                for aP in range(NA):
                    if Wrep[aP] <= -1e17:
                        continue
                    w = math.exp((Wrep[aP]-mW)/rhoA)
                    S += w
                    ibPp = bprime_of_a[aP]
                    bPp = b_grid[ibPp]
                    _, EV_P_cont, _, EmupayP = exp_next_objects(
                        izz, 1, 0, 0, 0, 0, ibPp,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NP = betaL*EmupayP
                    qP = solve_price_single(bP, bPp, NP)
                    Eqb += w*(qP*bPp)
                Eqb = Eqb/S if S>0 else 0.0

            CL = yL + (1.0-dP)*bP - (1.0-dP)*Eqb
            if CL <= 0.01:
                CL = 0.01
            mu_tilde[sid] = CL**(-sigmaL)
            continue

        # r=1 joint access
        VPrep_cond = np.empty(NA)
        VPdef_cond = np.empty(NA)
        dP_cond = np.empty(NA)

        EqOb_RR = np.zeros(NA)
        EqPb_RR = np.zeros(NA)
        EEV_O_RR = np.zeros(NA)

        for aO in range(NA):
            ibOp = bprime_of_a[aO]
            itauO = tau_of_a[aO]
            bOp = b_grid[ibOp]

            Wp = np.empty(NA)
            for aP in range(NA):
                ibPp = bprime_of_a[aP]
                itauP = tau_of_a[aP]
                if ok_rep[izP,itauP] == 0:
                    Wp[aP] = -1e18
                    continue
                bPp = b_grid[ibPp]

                EV_O_cont, EV_P_cont, EmupayO, EmupayP = exp_next_objects(
                    izz, 0, 0, 0, 0, ibOp, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NO = betaL*EmupayO
                NP = betaL*EmupayP

                qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)

                gP = rev_rep[izP,itauP] + qP*bPp - bP
                if gP <= 0.0:
                    Wp[aP] = -1e18
                    continue
                Wp[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont

            VrepP = ev_logsumexp(Wp, rhoA)

            mWp = -1e30
            for aP in range(NA):
                if Wp[aP] > mWp:
                    mWp = Wp[aP]
            Sp = 0.0
            EqO = 0.0
            EqP = 0.0
            EEV = 0.0
            if mWp > -1e20:
                for aP in range(NA):
                    if Wp[aP] <= -1e17:
                        continue
                    w = math.exp((Wp[aP]-mWp)/rhoA)
                    Sp += w
                    ibPp = bprime_of_a[aP]
                    bPp = b_grid[ibPp]

                    EV_O_cont, EV_P_cont, EmupayO, EmupayP = exp_next_objects(
                        izz, 0, 0, 0, 0, ibOp, ibPp,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NO = betaL*EmupayO
                    NP = betaL*EmupayP
                    qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)

                    EqO += w*(qO*bOp)
                    EqP += w*(qP*bPp)
                    EEV += w*(EV_O_cont)
                EqO = EqO/Sp if Sp>0 else 0.0
                EqP = EqP/Sp if Sp>0 else 0.0
                EEV = EEV/Sp if Sp>0 else 0.0

            EqOb_RR[aO] = EqO
            EqPb_RR[aO] = EqP
            EEV_O_RR[aO] = EEV
            VPrep_cond[aO] = VrepP

            Wdef_tauP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    Wdef_tauP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _ = exp_next_objects(
                    izz, 0, 0, 0, 1, ibOp, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tauP[it] = u + v_g(g) + beta*EV_P_cont
            VdefP = ev_logsumexp(Wdef_tauP, rhotau)
            VPdef_cond[aO] = VdefP
            dP_cond[aO] = logit_prob(VdefP, VrepP, rhoD)

        W_O_rep = np.empty(NA)
        EqO_total = np.empty(NA)
        EqP_total = np.empty(NA)

        for aO in range(NA):
            ibOp = bprime_of_a[aO]
            itauO = tau_of_a[aO]
            if ok_rep[izO,itauO] == 0:
                W_O_rep[aO] = -1e18
                EqO_total[aO] = 0.0
                EqP_total[aO] = 0.0
                continue
            bOp = b_grid[ibOp]

            EV_O_cont_RD, _, EmupayO_RD, _ = exp_next_objects(
                izz, 0, 0, 0, 1, ibOp, 0,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            NO_RD = betaL*EmupayO_RD
            qO_RD = solve_price_single(bO, bOp, NO_RD)

            dP_here = dP_cond[aO]
            EqO = (1.0-dP_here)*EqOb_RR[aO] + dP_here*(qO_RD*bOp)
            EqP = (1.0-dP_here)*EqPb_RR[aO] + dP_here*0.0
            EV_O_cont = (1.0-dP_here)*EEV_O_RR[aO] + dP_here*EV_O_cont_RD

            EqO_total[aO] = EqO
            EqP_total[aO] = EqP

            gO = rev_rep[izO,itauO] + EqO - bO
            if gO <= 0.0:
                W_O_rep[aO] = -1e18
                continue
            W_O_rep[aO] = u_rep[izO,itauO] + v_g(gO) + beta*EV_O_cont

        VrepO = ev_logsumexp(W_O_rep, rhoA)

        WrepP_dO1 = np.empty(NA)
        for aP in range(NA):
            ibPp = bprime_of_a[aP]
            itauP = tau_of_a[aP]
            if ok_rep[izP,itauP] == 0:
                WrepP_dO1[aP] = -1e18
                continue
            bPp = b_grid[ibPp]
            _, EV_P_cont, _, EmupayP = exp_next_objects(
                izz, 0, 0, 1, 0, 0, ibPp,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            NP = betaL*EmupayP
            qP = solve_price_single(bP, bPp, NP)
            gP = rev_rep[izP,itauP] + qP*bPp - bP
            if gP <= 0.0:
                WrepP_dO1[aP] = -1e18
                continue
            WrepP_dO1[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont
        VrepP_dO1 = ev_logsumexp(WrepP_dO1, rhoA)

        WdefP_tau_dO1 = np.empty(NT)
        for it in range(NT):
            if ok_defA[izP,it] == 0:
                WdefP_tau_dO1[it] = -1e18
                continue
            g = rev_defA[izP,it]
            u = u_defA[izP,it]
            _, EV_P_cont, _, _ = exp_next_objects(
                izz, 0, 0, 1, 1, 0, 0,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            WdefP_tau_dO1[it] = u + v_g(g) + beta*EV_P_cont
        VdefP_dO1 = ev_logsumexp(WdefP_tau_dO1, rhotau)
        dP_dO1 = logit_prob(VdefP_dO1, VrepP_dO1, rhoD)

        WdefO_tau = np.empty(NT)
        for it in range(NT):
            if ok_defA[izO,it] == 0:
                WdefO_tau[it] = -1e18
                continue
            g = rev_defA[izO,it]
            u = u_defA[izO,it]
            EV_O_cont_repP = exp_next_objects(izz, 0, 0, 1, 0, 0, 0,
                                              V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)[0]
            EV_O_cont_defP = exp_next_objects(izz, 0, 0, 1, 1, 0, 0,
                                              V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)[0]
            EV_O_cont = (1.0-dP_dO1)*EV_O_cont_repP + dP_dO1*EV_O_cont_defP
            WdefO_tau[it] = u + v_g(g) + beta*EV_O_cont
        VdefO = ev_logsumexp(WdefO_tau, rhotau)

        dO = logit_prob(VdefO, VrepO, rhoD)
        VO = inclusive_defrep(VdefO, VrepO, rhoD)

        mWO = -1e30
        for aO in range(NA):
            if W_O_rep[aO] > mWO:
                mWO = W_O_rep[aO]
        SO = 0.0
        EP_Vrep = 0.0
        EP_Vdef = 0.0
        EqO_rep = 0.0
        EqP_rep = 0.0
        if mWO > -1e20:
            for aO in range(NA):
                if W_O_rep[aO] <= -1e17:
                    continue
                w = math.exp((W_O_rep[aO]-mWO)/rhoA)
                SO += w
                EP_Vrep += w*VPrep_cond[aO]
                EP_Vdef += w*VPdef_cond[aO]
                EqO_rep += w*EqO_total[aO]
                EqP_rep += w*EqP_total[aO]
            EP_Vrep = EP_Vrep/SO if SO>0 else -1e18
            EP_Vdef = EP_Vdef/SO if SO>0 else -1e18
            EqO_rep = EqO_rep/SO if SO>0 else 0.0
            EqP_rep = EqP_rep/SO if SO>0 else 0.0

        VrepP_eq = (1.0-dO)*EP_Vrep + dO*VrepP_dO1
        VdefP_eq = (1.0-dO)*EP_Vdef + dO*VdefP_dO1
        dP = logit_prob(VdefP_eq, VrepP_eq, rhoD)
        VP = inclusive_defrep(VdefP_eq, VrepP_eq, rhoD)

        V_O_new[sid] = VO
        V_P_new[sid] = VP
        d_O_new[sid] = dO
        d_P_new[sid] = dP

        mWp = -1e30
        for aP in range(NA):
            if WrepP_dO1[aP] > mWp:
                mWp = WrepP_dO1[aP]
        Sp = 0.0
        EqPb_dO1 = 0.0
        if mWp > -1e20:
            for aP in range(NA):
                if WrepP_dO1[aP] <= -1e17:
                    continue
                w = math.exp((WrepP_dO1[aP]-mWp)/rhoA)
                Sp += w
                ibPp = bprime_of_a[aP]
                bPp = b_grid[ibPp]
                _, _, _, EmupayP2 = exp_next_objects(
                    izz, 0, 0, 1, 0, 0, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NP2 = betaL*EmupayP2
                qP2 = solve_price_single(bP, bPp, NP2)
                EqPb_dO1 += w*(qP2*bPp)
            EqPb_dO1 = EqPb_dO1/Sp if Sp>0 else 0.0

        Erep = (1.0-dO)*bO + (1.0-dP)*bP
        Epurch = (1.0-dO)*(EqO_rep + EqP_rep) + dO*(1.0-dP_dO1)*EqPb_dO1
        CL = yL + Erep - Epurch
        if CL <= 0.01:
            CL = 0.01
        mu_tilde[sid] = CL**(-sigmaL)

    return V_O_new, V_P_new, d_O_new, d_P_new, mu_tilde

# Solve fixed point
max_iter = 300
tol = 1e-5

print("\nStarting fixed point iteration...", flush=True)
for it in range(max_iter):
    V_O_new, V_P_new, d_O_new, d_P_new, mu_tilde = fixed_point_update(V_O, V_P, d_O, d_P, muL, sid_of)

    # Clamp mu_tilde to prevent numerical explosion
    mu_max = 50.0
    mu_tilde = np.minimum(mu_tilde, mu_max)

    muL_new = (1.0-xi_q)*muL + xi_q*mu_tilde
    V_O_upd = (1.0-xi_V)*V_O + xi_V*V_O_new
    V_P_upd = (1.0-xi_V)*V_P + xi_V*V_P_new
    d_O_upd = (1.0-xi_V)*d_O + xi_V*d_O_new
    d_P_upd = (1.0-xi_V)*d_P + xi_V*d_P_new

    err = max(
        np.max(np.abs(muL_new - muL)),
        np.max(np.abs(V_O_upd - V_O)),
        np.max(np.abs(V_P_upd - V_P)),
        np.max(np.abs(d_O_upd - d_O)),
        np.max(np.abs(d_P_upd - d_P)),
    )

    muL = muL_new
    V_O, V_P, d_O, d_P = V_O_upd, V_P_upd, d_O_upd, d_P_upd

    if (it+1) % 20 == 0 or it == 0:
        print(f"iter {it+1:3d} | sup err {err:.3e} | mean dO {d_O.mean():.3f} mean dP {d_P.mean():.3f}", flush=True)
    if err < tol:
        print("Converged.", flush=True)
        break

V_O_star = V_O.copy()
V_P_star = V_P.copy()
d_O_star = d_O.copy()
d_P_star = d_P.copy()
muL_star = muL.copy()

print("\nFixed point iteration complete.", flush=True)

# ============================================================
# Compute bond price function for O
# Fix income at middle z for both countries
# Plot qO as function of b'_O for:
#   - Joint access (P has market access) - Blue
#   - Sole O (P excluded) - Orange
# ============================================================

print("\nComputing price functions...", flush=True)

# Middle z indices
izO_mid = NZ // 2
izP_mid = NZ // 2
izz_mid = izO_mid + NZ * izP_mid  # joint index for (zO_mid, zP_mid)

# Fix current debt at 0 for both countries (or some other value)
ibO_current = 0
ibP_current = 0
bO_current = b_grid[ibO_current]
bP_current = b_grid[ibP_current]

# Create finer grid for b'_O for plotting
N_plot = 50
bprime_O_plot = np.linspace(0.01, bmax, N_plot)  # start from small positive to avoid degenerate q at b'=0

# Storage for prices
qO_joint = np.zeros(N_plot)   # P has market access (joint)
qO_sole = np.zeros(N_plot)    # P has no market access (sole O)

# For joint access, we need to integrate over P's behavior
# For simplicity, use a representative tau and average over P's actions

# Reference tau (middle of grid)
itau_ref = NT // 2

# Compute prices
for i, bOp in enumerate(bprime_O_plot):
    # Find closest grid point for b'_O (for expectations)
    ibOp_nearest = int(np.argmin(np.abs(b_grid - bOp)))
    
    # === Sole O case (r=2): single issuer pricing ===
    # Price: qO * CL^{-sigmaL} = N_O where N_O = betaL * E[mu' * (1-d')]
    EV_O_cont, _, EmupayO, _ = exp_next_objects(
        izz_mid, 0, 1, 0, 0, ibOp_nearest, 0,
        V_O_star, V_P_star, d_O_star, d_P_star, muL_star, sid_of
    )
    NO = betaL * EmupayO
    qO_sole[i] = solve_price_single(bO_current, bOp, NO)
    
    # === Joint access case (r=1): RR pricing with P behavior ===
    # Average over P's possible actions, weighted by P's policy
    # For simplicity, compute the average qO across P's debt choices
    
    qO_sum = 0.0
    weight_sum = 0.0
    
    for ibPp in range(NB):
        bPp = b_grid[ibPp]
        
        EV_O_cont, EV_P_cont, EmupayO, EmupayP = exp_next_objects(
            izz_mid, 0, 0, 0, 0, ibOp_nearest, ibPp,
            V_O_star, V_P_star, d_O_star, d_P_star, muL_star, sid_of
        )
        NO = betaL * EmupayO
        NP = betaL * EmupayP
        
        qO_RR, qP_RR = solve_price_RR(bO_current, bP_current, bOp, bPp, NO, NP)
        
        # Use uniform weights for simplicity (or could weight by P's policy)
        qO_sum += qO_RR
        weight_sum += 1.0
    
    qO_joint[i] = qO_sum / weight_sum if weight_sum > 0 else 0.0

# ============================================================
# Create the plot
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(bprime_O_plot, qO_joint, 'b-', linewidth=2.5, label='P has market access (joint)')
plt.plot(bprime_O_plot, qO_sole, color='orange', linestyle='-', linewidth=2.5, label='P has no market access (sole O)')

plt.xlabel("Debt choice $b'_O$", fontsize=14)
plt.ylabel("Bond price $q_O$", fontsize=14)
plt.title("Bond Price Function for O\n(Income levels held constant at middle grid point)", fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.xlim([0, bmax])
plt.ylim([0, 1.1])

# Add annotations
plt.axhline(y=betaL, color='gray', linestyle='--', alpha=0.5, label=f'Risk-free price = {betaL}')
plt.text(bmax*0.7, betaL+0.02, f'Risk-free: {betaL}', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('/workspace/price_function_O.png', dpi=150)
print("\nPlot saved to /workspace/price_function_O.png", flush=True)

# Also show some numerical details
print("\n" + "="*60, flush=True)
print("PRICE FUNCTION DETAILS", flush=True)
print("="*60, flush=True)
print(f"Income level (z_O, z_P): ({z_grid[izO_mid]:.4f}, {z_grid[izP_mid]:.4f})", flush=True)
print(f"Current debt (b_O, b_P): ({bO_current:.4f}, {bP_current:.4f})", flush=True)
print(f"\nb'_O\t\tq_O (joint)\tq_O (sole)", flush=True)
print("-"*50, flush=True)
for i in range(0, N_plot, 5):
    print(f"{bprime_O_plot[i]:.4f}\t\t{qO_joint[i]:.4f}\t\t{qO_sole[i]:.4f}", flush=True)

plt.show()
