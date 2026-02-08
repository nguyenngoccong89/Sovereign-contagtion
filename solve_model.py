# ============================================================
# Solve and simulate model01 - Sovereign debt with two countries
# ============================================================

# Setup
import subprocess
import sys

# Install numba if not present
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "numba"])

import numpy as np
import math
from numba import njit, prange
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# Parameters (exactly as your appendix + your grid/sim settings)
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
NZ = 7  # Original: 7 (use 5 for faster testing)
tauchen_m = 3.0

# Default cap rule
delta = 0.90

# Market access
lam = 0.20  # Pr(m'=0|m=1)

# Lender
sigmaL = 2.5
betaL  = 0.98
yL     = 0.80

# Grids requested (original: 11; use 7 for faster testing)
NB = 11
NT = 11
bmax = 0.8
taumax = 0.8

# Simulation requested (original: T=2000, N=1000, burnin=500)
T_sim = 2000
N_sim = 1000
burnin = 500

# EV + damping (algorithmic parameters you said you use; not economics)
rhoA   = 0.02
rhotau = 0.02
rhoD   = 0.02
xi_q   = 0.5   # inner MU damping
xi_V   = 0.5   # outer value/default damping (stabilizing)

EULER_GAMMA = 0.5772156649015329

# ============================================================
# Tauchen discretization (log AR(1)) and stationary dist
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
z_def = np.minimum(z_grid, delta * zbar)  # EXACT: z_def = min{z, delta*zbar}

print("z_grid:", z_grid, flush=True)
print("zbar:", zbar, flush=True)
print("z_def:", z_def, flush=True)
print("Row sums Pz:", Pz.sum(axis=1), flush=True)

# ============================================================
# Joint transition for (zO,zP) and index maps
# ============================================================

Pzz = np.kron(Pz, Pz)     # size NZ^2 x NZ^2
NZ2 = NZ*NZ

izO_of_zz = np.empty(NZ2, dtype=np.int64)
izP_of_zz = np.empty(NZ2, dtype=np.int64)
k = 0
for izP in range(NZ):
    for izO in range(NZ):
        izO_of_zz[k] = izO
        izP_of_zz[k] = izP
        k += 1

Pzz_cum = np.cumsum(Pzz, axis=1)

print("Pzz shape:", Pzz.shape, flush=True)

# ============================================================
# Debt/tax grids and action mappings a=(b',tau)
# ============================================================

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
# Household block (exact) + utility and feasibility
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

# Precompute for repay (z) and default/exclusion (z_def)
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
# State space with regime restriction + sid lookup
# Regimes r:
#  1:(0,0) joint access
#  2:(0,1) only O access
#  3:(1,0) only P access
#  4:(1,1) both excluded
# with restriction: if m_i=1 then b_i=0
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

# sid_of[izz, ibO, ibP, r] -> sid (r=1..4)
sid_of = -np.ones((NZ2, NB, NB, 5), dtype=np.int64)
for sid in range(NS):
    sid_of[state_zz[sid], state_ibO[sid], state_ibP[sid], state_r[sid]] = sid

print("NS:", NS, flush=True)

# ============================================================
# EV operators (exact) and default logit (exact)
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

# ============================================================
# Expectations (Step 2) and price solvers (exact)
# ============================================================

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
    """
    EXACT Step 2 expectation operator:
    loops over z' and re-entry lotteries for excluded countries.
    Returns EV_O, EV_P, EpayO, EpayP, Emu.
    """
    EV_O = 0.0
    EV_P = 0.0
    EpayO = 0.0
    EpayP = 0.0
    Emu   = 0.0

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
                EpayO += prob * (1.0 - d_O[sid2])
                EpayP += prob * (1.0 - d_P[sid2])
                Emu   += prob * muL[sid2]

    return EV_O, EV_P, EpayO, EpayP, Emu

@njit
def solve_price_single(b_i, bprime_i, N_i):
    # solve q * (yL + b_i - q*bprime_i)^(-sigmaL) = N_i
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
    # qP = qO*(NP/NO); solve qO*(yL+bO+bP-qO*bO'-qP*bP')^{-sigmaL} = NO
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

print("Compiled expectations + price solvers.", flush=True)

# ============================================================
# Stationary equilibrium solver (exact algorithmic structure)
# - EV over actions and tau
# - logit default
# - lender MU fixed point with damping (xi_q)
# - outer damping for values/defaults (xi_V)
# ============================================================

# Initialize guesses
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

        # ------------------------------------------------------------
        # r=4 both excluded: tau EV only, d=0
        # ------------------------------------------------------------
        if r == 4:
            # O excluded
            valsO = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    valsO[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _, _ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsO[it] = u + v_g(g) + beta*EV_O_cont
            VO = ev_logsumexp(valsO, rhotau)

            # P excluded
            valsP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    valsP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _, _ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0,
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

        # ------------------------------------------------------------
        # r=2 sole O (P excluded)
        # ------------------------------------------------------------
        if r == 2:
            # P excluded
            valsP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    valsP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _, _ = exp_next_objects(izz, 0, 1, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsP[it] = u + v_g(g) + beta*EV_P_cont
            V_P_new[sid] = ev_logsumexp(valsP, rhotau)
            d_P_new[sid] = 0.0

            # O repay EV over aO
            Wrep = np.empty(NA)
            for aO in range(NA):
                ibOp = bprime_of_a[aO]
                itauO = tau_of_a[aO]
                if ok_rep[izO,itauO] == 0:
                    Wrep[aO] = -1e18
                    continue
                bOp = b_grid[ibOp]

                EV_O_cont, _, EpayO, _, Emu = exp_next_objects(
                    izz, 0, 1, 0, 0, ibOp, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NO = betaL*(Emu*EpayO)
                qO = solve_price_single(bO, bOp, NO)

                gO = rev_rep[izO,itauO] + qO*bOp - bO
                if gO <= 0.0:
                    Wrep[aO] = -1e18
                    continue
                Wrep[aO] = u_rep[izO,itauO] + v_g(gO) + beta*EV_O_cont

            VrepO = ev_logsumexp(Wrep, rhoA)

            # O default EV over tau
            Wdef_tau = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    Wdef_tau[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _, _ = exp_next_objects(
                    izz, 0, 1, 1, 0, 0, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tau[it] = u + v_g(g) + beta*EV_O_cont

            VdefO = ev_logsumexp(Wdef_tau, rhotau)
            dO = logit_prob(VdefO, VrepO, rhoD)
            VO = inclusive_defrep(VdefO, VrepO, rhoD)

            V_O_new[sid] = VO
            d_O_new[sid] = dO

            # Lender exact expected consumption in this state:
            # purchases use π(aO|repay) from Wrep
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
                    EV_O_cont, _, EpayO, _, Emu = exp_next_objects(
                        izz, 0, 1, 0, 0, ibOp, 0,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NO = betaL*(Emu*EpayO)
                    qO = solve_price_single(bO, bOp, NO)
                    Eqb += w*(qO*bOp)
                Eqb = Eqb/S if S>0 else 0.0

            CL = yL + (1.0-dO)*bO - (1.0-dO)*Eqb
            if CL <= 1e-14:
                CL = 1e-14
            mu_tilde[sid] = CL**(-sigmaL)
            continue

        # ------------------------------------------------------------
        # r=3 sole P (O excluded)
        # ------------------------------------------------------------
        if r == 3:
            # O excluded
            valsO = np.empty(NT)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    valsO[it] = -1e18
                    continue
                g = rev_defA[izO,it]
                u = u_defA[izO,it]
                EV_O_cont, _, _, _, _ = exp_next_objects(izz, 1, 0, 0, 0, 0, 0,
                                                         V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_)
                valsO[it] = u + v_g(g) + beta*EV_O_cont
            V_O_new[sid] = ev_logsumexp(valsO, rhotau)
            d_O_new[sid] = 0.0

            # P repay EV over aP
            Wrep = np.empty(NA)
            for aP in range(NA):
                ibPp = bprime_of_a[aP]
                itauP = tau_of_a[aP]
                if ok_rep[izP,itauP] == 0:
                    Wrep[aP] = -1e18
                    continue
                bPp = b_grid[ibPp]

                _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
                    izz, 1, 0, 0, 0, 0, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NP = betaL*(Emu*EpayP)
                qP = solve_price_single(bP, bPp, NP)

                gP = rev_rep[izP,itauP] + qP*bPp - bP
                if gP <= 0.0:
                    Wrep[aP] = -1e18
                    continue
                Wrep[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont

            VrepP = ev_logsumexp(Wrep, rhoA)

            # P default EV over tau
            Wdef_tau = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    Wdef_tau[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _, _ = exp_next_objects(
                    izz, 1, 0, 0, 1, 0, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tau[it] = u + v_g(g) + beta*EV_P_cont

            VdefP = ev_logsumexp(Wdef_tau, rhotau)
            dP = logit_prob(VdefP, VrepP, rhoD)
            VP = inclusive_defrep(VdefP, VrepP, rhoD)

            V_P_new[sid] = VP
            d_P_new[sid] = dP

            # Lender exact expected consumption:
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
                    _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
                        izz, 1, 0, 0, 0, 0, ibPp,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NP = betaL*(Emu*EpayP)
                    qP = solve_price_single(bP, bPp, NP)
                    Eqb += w*(qP*bPp)
                Eqb = Eqb/S if S>0 else 0.0

            CL = yL + (1.0-dP)*bP - (1.0-dP)*Eqb
            if CL <= 1e-14:
                CL = 1e-14
            mu_tilde[sid] = CL**(-sigmaL)
            continue

        # ------------------------------------------------------------
        # r=1 joint access: Stackelberg with EV (exactly your Step 3–5)
        # ------------------------------------------------------------
        # Follower conditional on leader action aO:
        VPrep_cond = np.empty(NA)
        VPdef_cond = np.empty(NA)
        dP_cond = np.empty(NA)

        # For leader continuation and purchases, we need expectations under πP_RR:
        EqOb_RR = np.zeros(NA)     # E[qO*bO' | P repays, given aO]
        EqPb_RR = np.zeros(NA)     # E[qP*bP' | P repays, given aO]
        EEV_O_RR = np.zeros(NA)    # E[ EV_O_next | P repays, given aO ]

        for aO in range(NA):
            ibOp = bprime_of_a[aO]
            itauO = tau_of_a[aO]
            bOp = b_grid[ibOp]

            # P repay: EV over aP with RR prices
            Wp = np.empty(NA)
            for aP in range(NA):
                ibPp = bprime_of_a[aP]
                itauP = tau_of_a[aP]
                if ok_rep[izP,itauP] == 0:
                    Wp[aP] = -1e18
                    continue
                bPp = b_grid[ibPp]

                EV_O_cont, EV_P_cont, EpayO, EpayP, Emu = exp_next_objects(
                    izz, 0, 0, 0, 0, ibOp, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NO = betaL*(Emu*EpayO)
                NP = betaL*(Emu*EpayP)

                qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)

                gP = rev_rep[izP,itauP] + qP*bPp - bP
                if gP <= 0.0:
                    Wp[aP] = -1e18
                    continue
                Wp[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont

            VrepP = ev_logsumexp(Wp, rhoA)

            # Build πP_RR weights for expectations used by leader
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

                    EV_O_cont, EV_P_cont, EpayO, EpayP, Emu = exp_next_objects(
                        izz, 0, 0, 0, 0, ibOp, ibPp,
                        V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                    )
                    NO = betaL*(Emu*EpayO)
                    NP = betaL*(Emu*EpayP)
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

            # P default EV over tau (conditional on leader repay aO)
            Wdef_tauP = np.empty(NT)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    Wdef_tauP[it] = -1e18
                    continue
                g = rev_defA[izP,it]
                u = u_defA[izP,it]
                _, EV_P_cont, _, _, _ = exp_next_objects(
                    izz, 0, 0, 0, 1, ibOp, 0,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                Wdef_tauP[it] = u + v_g(g) + beta*EV_P_cont
            VdefP = ev_logsumexp(Wdef_tauP, rhotau)
            VPdef_cond[aO] = VdefP
            dP_cond[aO] = logit_prob(VdefP, VrepP, rhoD)

        # Leader repay: EV over aO anticipating follower behavior
        W_O_rep = np.empty(NA)
        # for lender purchases later: need expected purchases under πO (repay)
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

            # RD branch (P defaults): single issuer price for O
            EV_O_cont_RD, _, EpayO_RD, _, Emu_RD = exp_next_objects(
                izz, 0, 0, 0, 1, ibOp, 0,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            NO_RD = betaL*(Emu_RD*EpayO_RD)
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

        # O default branch: need P behavior when dO=1 (single issuer P today)
        # P repay EV (single issuer)
        WrepP_dO1 = np.empty(NA)
        for aP in range(NA):
            ibPp = bprime_of_a[aP]
            itauP = tau_of_a[aP]
            if ok_rep[izP,itauP] == 0:
                WrepP_dO1[aP] = -1e18
                continue
            bPp = b_grid[ibPp]
            _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
                izz, 0, 0, 1, 0, 0, ibPp,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            NP = betaL*(Emu*EpayP)
            qP = solve_price_single(bP, bPp, NP)
            gP = rev_rep[izP,itauP] + qP*bPp - bP
            if gP <= 0.0:
                WrepP_dO1[aP] = -1e18
                continue
            WrepP_dO1[aP] = u_rep[izP,itauP] + v_g(gP) + beta*EV_P_cont
        VrepP_dO1 = ev_logsumexp(WrepP_dO1, rhoA)

        # P default EV over tau when dO=1
        WdefP_tau_dO1 = np.empty(NT)
        for it in range(NT):
            if ok_defA[izP,it] == 0:
                WdefP_tau_dO1[it] = -1e18
                continue
            g = rev_defA[izP,it]
            u = u_defA[izP,it]
            _, EV_P_cont, _, _, _ = exp_next_objects(
                izz, 0, 0, 1, 1, 0, 0,
                V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
            )
            WdefP_tau_dO1[it] = u + v_g(g) + beta*EV_P_cont
        VdefP_dO1 = ev_logsumexp(WdefP_tau_dO1, rhotau)
        dP_dO1 = logit_prob(VdefP_dO1, VrepP_dO1, rhoD)

        # O default EV over tau, continuation integrates over P default under dO=1
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

        # Equilibrium P: integrate over leader repay actions with πO + leader default prob
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

        # Lender expected consumption in joint access (exact Step 10 with EV integration)
        # repayments: (1-dO)*bO + (1-dP)*bP
        # purchases:
        #  - if O repays: buy O and (if P repays) P, summarized by EqO_rep+EqP_rep
        #  - if O defaults: buy P if it repays under dO=1; compute expected qP*bP' from WrepP_dO1
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
                _, _, _, EpayP2, Emu2 = exp_next_objects(
                    izz, 0, 0, 1, 0, 0, ibPp,
                    V_O_old, V_P_old, d_O_old, d_P_old, muL_old, sid_of_
                )
                NP2 = betaL*(Emu2*EpayP2)
                qP2 = solve_price_single(bP, bPp, NP2)
                EqPb_dO1 += w*(qP2*bPp)
            EqPb_dO1 = EqPb_dO1/Sp if Sp>0 else 0.0

        Erep = (1.0-dO)*bO + (1.0-dP)*bP
        Epurch = (1.0-dO)*(EqO_rep + EqP_rep) + dO*(1.0-dP_dO1)*EqPb_dO1
        CL = yL + Erep - Epurch
        if CL <= 1e-14:
            CL = 1e-14
        mu_tilde[sid] = CL**(-sigmaL)

    return V_O_new, V_P_new, d_O_new, d_P_new, mu_tilde

# Solve fixed point
max_iter = 200
tol = 1e-5

print("\nStarting fixed point iteration...", flush=True)
for it in range(max_iter):
    V_O_new, V_P_new, d_O_new, d_P_new, mu_tilde = fixed_point_update(V_O, V_P, d_O, d_P, muL, sid_of)

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

    if (it+1) % 5 == 0 or it == 0:
        print(f"iter {it+1:3d} | sup err {err:.3e} | mean dO {d_O.mean():.3f} mean dP {d_P.mean():.3f}", flush=True)
    if err < tol:
        print("Converged.", flush=True)
        break

V_O_star, V_P_star, d_O_star, d_P_star, muL_star = V_O.copy(), V_P.copy(), d_O.copy(), d_P.copy(), muL.copy()

# ============================================================
# FULL exact conditional-joint policy precompute
# ============================================================

# Identify state subsets
joint_sids = np.where(state_r == 1)[0]
soleO_sids = np.where(state_r == 2)[0]
soleP_sids = np.where(state_r == 3)[0]
excl_sids  = np.where(state_r == 4)[0]

NS1 = joint_sids.size
NS2 = soleO_sids.size
NS3 = soleP_sids.size
NS4 = excl_sids.size
print("State counts: joint", NS1, "soleO", NS2, "soleP", NS3, "excl", NS4, flush=True)

# sid -> index maps
joint_index_of_sid = -np.ones(NS, dtype=np.int64)
for j,sid in enumerate(joint_sids): joint_index_of_sid[sid]=j
soleO_index_of_sid = -np.ones(NS, dtype=np.int64)
for j,sid in enumerate(soleO_sids): soleO_index_of_sid[sid]=j
soleP_index_of_sid = -np.ones(NS, dtype=np.int64)
for j,sid in enumerate(soleP_sids): soleP_index_of_sid[sid]=j

def softmax_EV(W, rho):
    m = np.max(W)
    if m < -1e17:
        return np.zeros_like(W, dtype=np.float64)
    w = np.exp((W - m) / rho)
    s = w.sum()
    if s <= 0:
        return np.zeros_like(W, dtype=np.float64)
    return w / s

def softmax_tau(W, rho):
    m = np.max(W)
    if m < -1e17:
        p = np.zeros_like(W, dtype=np.float64); p[0]=1.0
        return p
    w = np.exp((W - m) / rho)
    s = w.sum()
    if s <= 0:
        p = np.zeros_like(W, dtype=np.float64); p[0]=1.0
        return p
    return w / s

# Allocate (float32 to fit memory)
piO_joint = np.zeros((NS1, NA), dtype=np.float32)
dP_cond   = np.zeros((NS1, NA), dtype=np.float32)
piP_RR    = np.zeros((NS1, NA, NA), dtype=np.float32)

dP_dO1    = np.zeros(NS1, dtype=np.float32)
piP_DR    = np.zeros((NS1, NA), dtype=np.float32)

piTau_O_def_joint = np.zeros((NS1, NT), dtype=np.float32)
piTau_P_def_joint = np.zeros((NS1, NA, NT), dtype=np.float32)
piTau_P_def_dO1   = np.zeros((NS1, NT), dtype=np.float32)

piO_sole = np.zeros((NS2, NA), dtype=np.float32)
piP_sole = np.zeros((NS3, NA), dtype=np.float32)
piTau_O_def_sole = np.zeros((NS2, NT), dtype=np.float32)
piTau_P_def_sole = np.zeros((NS3, NT), dtype=np.float32)

# Shorthands
V_O = V_O_star; V_P = V_P_star; d_O = d_O_star; d_P = d_P_star; muL = muL_star

print("\nComputing joint access policies...", flush=True)
# --- joint access policies ---
for js, sid in enumerate(joint_sids):
    izz = int(state_zz[sid])
    izO = int(izO_of_zz[izz]); izP = int(izP_of_zz[izz])
    ibO = int(state_ibO[sid]); ibP = int(state_ibP[sid])
    bO = float(b_grid[ibO]);   bP = float(b_grid[ibP])

    # follower conditional on leader action aO
    for aO in range(NA):
        ibOp = int(bprime_of_a[aO]); itauO = int(tau_of_a[aO])
        bOp = float(b_grid[ibOp])

        # P repay indices W(aP)
        Wrep = np.full(NA, -1e18, dtype=np.float64)
        for aP in range(NA):
            ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
            if ok_rep[izP,itauP] == 0:
                continue
            bPp = float(b_grid[ibPp])

            EV_O_cont, EV_P_cont, EpayO, EpayP, Emu = exp_next_objects(
                izz, 0, 0, 0, 0, ibOp, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
            )
            NO = betaL*(Emu*EpayO)
            NP = betaL*(Emu*EpayP)
            qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)

            gP = float(rev_rep[izP,itauP]) + qP*bPp - bP
            if gP <= 0.0:
                continue
            Wrep[aP] = float(u_rep[izP,itauP]) + float(v_g(gP)) + beta*EV_P_cont

        pP = softmax_EV(Wrep, rhoA)
        piP_RR[js, aO, :] = pP.astype(np.float32)

        # P default tau indices
        Wtau = np.full(NT, -1e18, dtype=np.float64)
        for it in range(NT):
            if ok_defA[izP,it] == 0:
                continue
            g = float(rev_defA[izP,it])
            _, EV_P_cont, *_ = exp_next_objects(
                izz, 0, 0, 0, 1, ibOp, 0, V_O, V_P, d_O, d_P, muL, sid_of
            )
            Wtau[it] = float(u_defA[izP,it]) + float(v_g(g)) + beta*EV_P_cont
        piTau_P_def_joint[js, aO, :] = softmax_tau(Wtau, rhotau).astype(np.float32)

        # dP_cond(s|aO)
        Vrep_incl = float(ev_logsumexp(Wrep, rhoA))
        Vdef_incl = float(ev_logsumexp(Wtau, rhotau))
        dP_cond[js, aO] = float(logit_prob(Vdef_incl, Vrep_incl, rhoD))

    # P when O defaults today (dO=1): single issuer
    Wrep_DR = np.full(NA, -1e18, dtype=np.float64)
    for aP in range(NA):
        ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
        if ok_rep[izP,itauP] == 0:
            continue
        bPp = float(b_grid[ibPp])
        _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
            izz, 0, 0, 1, 0, 0, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
        )
        NP = betaL*(Emu*EpayP)
        qP = solve_price_single(bP, bPp, NP)
        gP = float(rev_rep[izP,itauP]) + qP*bPp - bP
        if gP <= 0.0:
            continue
        Wrep_DR[aP] = float(u_rep[izP,itauP]) + float(v_g(gP)) + beta*EV_P_cont

    piP_DR[js,:] = softmax_EV(Wrep_DR, rhoA).astype(np.float32)

    Wtau_dO1 = np.full(NT, -1e18, dtype=np.float64)
    for it in range(NT):
        if ok_defA[izP,it] == 0:
            continue
        g = float(rev_defA[izP,it])
        _, EV_P_cont, *_ = exp_next_objects(
            izz, 0, 0, 1, 1, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of
        )
        Wtau_dO1[it] = float(u_defA[izP,it]) + float(v_g(g)) + beta*EV_P_cont

    piTau_P_def_dO1[js,:] = softmax_tau(Wtau_dO1, rhotau).astype(np.float32)

    VrepP_dO1 = float(ev_logsumexp(Wrep_DR, rhoA))
    VdefP_dO1 = float(ev_logsumexp(Wtau_dO1, rhotau))
    dP_dO1[js] = float(logit_prob(VdefP_dO1, VrepP_dO1, rhoD))

    # Leader repay policy πO(aO|s, repay)
    W_O_rep = np.full(NA, -1e18, dtype=np.float64)
    for aO in range(NA):
        ibOp = int(bprime_of_a[aO]); itauO = int(tau_of_a[aO])
        if ok_rep[izO,itauO] == 0:
            continue
        bOp = float(b_grid[ibOp])

        # RD branch price for O
        EV_O_RD, _, EpayO_RD, _, Emu_RD = exp_next_objects(
            izz, 0, 0, 0, 1, ibOp, 0, V_O, V_P, d_O, d_P, muL, sid_of
        )
        NO_RD = betaL*(Emu_RD*EpayO_RD)
        qO_RD = solve_price_single(bO, bOp, NO_RD)

        # RR branch integrate over πP_RR
        pP = piP_RR[js, aO, :].astype(np.float64)
        EqOb_RR = 0.0
        EEV_O_RR = 0.0
        for aP in range(NA):
            pp = pP[aP]
            if pp == 0.0:
                continue
            ibPp = int(bprime_of_a[aP]); bPp = float(b_grid[ibPp])

            EV_O_cont, EV_P_cont, EpayO, EpayP, Emu = exp_next_objects(
                izz, 0, 0, 0, 0, ibOp, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
            )
            NO = betaL*(Emu*EpayO)
            NP = betaL*(Emu*EpayP)
            qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)
            EqOb_RR += pp*(qO*bOp)
            EEV_O_RR += pp*(EV_O_cont)

        dP_here = float(dP_cond[js, aO])
        EqOb = (1.0-dP_here)*EqOb_RR + dP_here*(qO_RD*bOp)
        EV_O_cont = (1.0-dP_here)*EEV_O_RR + dP_here*EV_O_RD

        gO = float(rev_rep[izO,itauO]) + EqOb - bO
        if gO <= 0.0:
            continue
        W_O_rep[aO] = float(u_rep[izO,itauO]) + float(v_g(gO)) + beta*EV_O_cont

    piO_joint[js,:] = softmax_EV(W_O_rep, rhoA).astype(np.float32)

    # O default tau policy (joint)
    WtauO = np.full(NT, -1e18, dtype=np.float64)
    for it in range(NT):
        if ok_defA[izO,it] == 0:
            continue
        g = float(rev_defA[izO,it])
        EV_O_cont_repP = exp_next_objects(izz, 0, 0, 1, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)[0]
        EV_O_cont_defP = exp_next_objects(izz, 0, 0, 1, 1, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)[0]
        EV_O_cont = (1.0-float(dP_dO1[js]))*EV_O_cont_repP + float(dP_dO1[js])*EV_O_cont_defP
        WtauO[it] = float(u_defA[izO,it]) + float(v_g(g)) + beta*EV_O_cont
    piTau_O_def_joint[js,:] = softmax_tau(WtauO, rhotau).astype(np.float32)

print("Joint policies done.", flush=True)

# ============================================================
# Sole borrower policies (exact)
# ============================================================

print("Computing sole borrower policies...", flush=True)
# Sole O (r=2)
for ks, sid in enumerate(soleO_sids):
    izz = int(state_zz[sid])
    izO = int(izO_of_zz[izz]); izP = int(izP_of_zz[izz])
    ibO = int(state_ibO[sid])
    bO = float(b_grid[ibO])

    # repay action policy
    W = np.full(NA, -1e18, dtype=np.float64)
    for aO in range(NA):
        ibOp = int(bprime_of_a[aO]); itauO = int(tau_of_a[aO])
        if ok_rep[izO,itauO] == 0:
            continue
        bOp = float(b_grid[ibOp])
        EV_O_cont, _, EpayO, _, Emu = exp_next_objects(
            izz, 0, 1, 0, 0, ibOp, 0, V_O, V_P, d_O, d_P, muL, sid_of
        )
        NO = betaL*(Emu*EpayO)
        qO = solve_price_single(bO, bOp, NO)
        gO = float(rev_rep[izO,itauO]) + qO*bOp - bO
        if gO <= 0.0:
            continue
        W[aO] = float(u_rep[izO,itauO]) + float(v_g(gO)) + beta*EV_O_cont
    piO_sole[ks,:] = softmax_EV(W, rhoA).astype(np.float32)

    # default tau policy
    Wtau = np.full(NT, -1e18, dtype=np.float64)
    for it in range(NT):
        if ok_defA[izO,it] == 0:
            continue
        g = float(rev_defA[izO,it])
        EV_O_cont = exp_next_objects(izz, 0, 1, 1, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)[0]
        Wtau[it] = float(u_defA[izO,it]) + float(v_g(g)) + beta*EV_O_cont
    piTau_O_def_sole[ks,:] = softmax_tau(Wtau, rhotau).astype(np.float32)

# Sole P (r=3)
for ks, sid in enumerate(soleP_sids):
    izz = int(state_zz[sid])
    izO = int(izO_of_zz[izz]); izP = int(izP_of_zz[izz])
    ibP = int(state_ibP[sid])
    bP = float(b_grid[ibP])

    W = np.full(NA, -1e18, dtype=np.float64)
    for aP in range(NA):
        ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
        if ok_rep[izP,itauP] == 0:
            continue
        bPp = float(b_grid[ibPp])
        _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
            izz, 1, 0, 0, 0, 0, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
        )
        NP = betaL*(Emu*EpayP)
        qP = solve_price_single(bP, bPp, NP)
        gP = float(rev_rep[izP,itauP]) + qP*bPp - bP
        if gP <= 0.0:
            continue
        W[aP] = float(u_rep[izP,itauP]) + float(v_g(gP)) + beta*EV_P_cont
    piP_sole[ks,:] = softmax_EV(W, rhoA).astype(np.float32)

    Wtau = np.full(NT, -1e18, dtype=np.float64)
    for it in range(NT):
        if ok_defA[izP,it] == 0:
            continue
        g = float(rev_defA[izP,it])
        EV_P_cont = exp_next_objects(izz, 1, 0, 0, 1, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)[1]
        Wtau[it] = float(u_defA[izP,it]) + float(v_g(g)) + beta*EV_P_cont
    piTau_P_def_sole[ks,:] = softmax_tau(Wtau, rhotau).astype(np.float32)

print("Sole borrower policies done.", flush=True)

# ============================================================
# EXACT simulation using FULL conditional-joint policies
# ============================================================

def draw_from_probs(p):
    u = np.random.rand()
    return int(np.searchsorted(np.cumsum(p), u, side="right"))

def draw_next_izz(cur_izz):
    u = np.random.rand()
    return int(np.searchsorted(Pzz_cum[cur_izz], u, side="right"))

def m_to_r(mO,mP):
    if mO==0 and mP==0: return 1
    if mO==0 and mP==1: return 2
    if mO==1 and mP==0: return 3
    return 4

def regime_to_m(r):
    if r==1: return (0,0)
    if r==2: return (0,1)
    if r==3: return (1,0)
    return (1,1)

# Storage post burn-in
T_eff = T_sim - burnin
paths = {
    "O_y": np.zeros((N_sim, T_eff)),
    "O_c": np.zeros((N_sim, T_eff)),
    "O_l": np.zeros((N_sim, T_eff)),
    "O_g": np.zeros((N_sim, T_eff)),
    "O_tau": np.zeros((N_sim, T_eff)),
    "O_b": np.zeros((N_sim, T_eff)),
    "O_q": np.zeros((N_sim, T_eff)),
    "O_def": np.zeros((N_sim, T_eff), dtype=np.int8),

    "P_y": np.zeros((N_sim, T_eff)),
    "P_c": np.zeros((N_sim, T_eff)),
    "P_l": np.zeros((N_sim, T_eff)),
    "P_g": np.zeros((N_sim, T_eff)),
    "P_tau": np.zeros((N_sim, T_eff)),
    "P_b": np.zeros((N_sim, T_eff)),
    "P_q": np.zeros((N_sim, T_eff)),
    "P_def": np.zeros((N_sim, T_eff), dtype=np.int8),

    "L_C": np.zeros((N_sim, T_eff)),
    "L_mu": np.zeros((N_sim, T_eff)),
}

# Initial state: middle z, zero debt, joint access
izz0 = (NZ//2) + NZ*(NZ//2)
sid0 = int(sid_of[izz0, 0, 0, 1])

print("\nStarting simulation...", flush=True)
for n in range(N_sim):
    if (n+1) % 100 == 0:
        print(f"  Simulating path {n+1}/{N_sim}", flush=True)
    sid = sid0
    for t in range(T_sim):
        izz = int(state_zz[sid])
        r   = int(state_r[sid])
        izO = int(izO_of_zz[izz]); izP = int(izP_of_zz[izz])
        ibO = int(state_ibO[sid]); ibP = int(state_ibP[sid])
        bO  = float(b_grid[ibO]);  bP  = float(b_grid[ibP])
        mO, mP = regime_to_m(r)

        # Today controls/outcomes
        dO_today = 0
        dP_today = 0
        ibOp = 0; itauO = 0; qO = 0.0
        ibPp = 0; itauP = 0; qP = 0.0

        if r == 1:
            js = int(joint_index_of_sid[sid])

            # leader default draw (exact from d_O_star(s))
            dO_today = 1 if np.random.rand() < float(d_O_star[sid]) else 0

            if dO_today == 0:
                # leader repay action draw
                aO = draw_from_probs(piO_joint[js,:])
                ibOp = int(bprime_of_a[aO]); itauO = int(tau_of_a[aO])
                bOp = float(b_grid[ibOp])

                # follower default conditional on (s,aO)
                dP_today = 1 if np.random.rand() < float(dP_cond[js,aO]) else 0

                if dP_today == 0:
                    # follower repay action draw conditional on (s,aO)
                    aP = draw_from_probs(piP_RR[js,aO,:])
                    ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
                    bPp = float(b_grid[ibPp])

                    # realized RR prices (exact)
                    EV_O_cont, EV_P_cont, EpayO, EpayP, Emu = exp_next_objects(
                        izz, 0, 0, 0, 0, ibOp, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
                    )
                    NO = betaL*(Emu*EpayO)
                    NP = betaL*(Emu*EpayP)
                    qO, qP = solve_price_RR(bO, bP, bOp, bPp, NO, NP)

                else:
                    # follower defaults: draw tau from π_tau_P_def_joint(s,aO)
                    itauP = draw_from_probs(piTau_P_def_joint[js,aO,:])
                    ibPp = 0; qP = 0.0
                    # O single issuer price qO_RD
                    EV_O_cont, _, EpayO, _, Emu = exp_next_objects(
                        izz, 0, 0, 0, 1, ibOp, 0, V_O, V_P, d_O, d_P, muL, sid_of
                    )
                    NO = betaL*(Emu*EpayO)
                    qO = solve_price_single(bO, bOp, NO)

            else:
                # leader defaults: draw tau from π_tau_O_def_joint(s)
                itauO = draw_from_probs(piTau_O_def_joint[js,:])
                ibOp = 0; qO = 0.0

                # follower behaves as sole issuer under dO=1
                dP_today = 1 if np.random.rand() < float(dP_dO1[js]) else 0
                if dP_today == 0:
                    aP = draw_from_probs(piP_DR[js,:])
                    ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
                    bPp = float(b_grid[ibPp])

                    _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
                        izz, 0, 0, 1, 0, 0, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
                    )
                    NP = betaL*(Emu*EpayP)
                    qP = solve_price_single(bP, bPp, NP)
                else:
                    itauP = draw_from_probs(piTau_P_def_dO1[js,:])
                    ibPp = 0; qP = 0.0

        elif r == 2:
            # Sole O
            ks = int(soleO_index_of_sid[sid])
            dO_today = 1 if np.random.rand() < float(d_O_star[sid]) else 0
            if dO_today == 0:
                aO = draw_from_probs(piO_sole[ks,:])
                ibOp = int(bprime_of_a[aO]); itauO = int(tau_of_a[aO])
                bOp = float(b_grid[ibOp])
                EV_O_cont, _, EpayO, _, Emu = exp_next_objects(
                    izz, 0, 1, 0, 0, ibOp, 0, V_O, V_P, d_O, d_P, muL, sid_of
                )
                NO = betaL*(Emu*EpayO)
                qO = solve_price_single(bO, bOp, NO)
            else:
                itauO = draw_from_probs(piTau_O_def_sole[ks,:])
                ibOp = 0; qO = 0.0

            # P excluded: (tau is only needed for macro series; exact EV-over-tau computed on the fly)
            WtauP = np.full(NT, -1e18, dtype=np.float64)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    continue
                g = float(rev_defA[izP,it])
                _, EV_P_cont, *_ = exp_next_objects(izz, 0, 1, 0, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)
                WtauP[it] = float(u_defA[izP,it]) + float(v_g(g)) + beta*EV_P_cont
            itauP = draw_from_probs(softmax_tau(WtauP, rhotau).astype(np.float32))
            dP_today = 0; ibPp = 0; qP = 0.0

        elif r == 3:
            # Sole P
            ks = int(soleP_index_of_sid[sid])
            dP_today = 1 if np.random.rand() < float(d_P_star[sid]) else 0
            if dP_today == 0:
                aP = draw_from_probs(piP_sole[ks,:])
                ibPp = int(bprime_of_a[aP]); itauP = int(tau_of_a[aP])
                bPp = float(b_grid[ibPp])
                _, EV_P_cont, _, EpayP, Emu = exp_next_objects(
                    izz, 1, 0, 0, 0, 0, ibPp, V_O, V_P, d_O, d_P, muL, sid_of
                )
                NP = betaL*(Emu*EpayP)
                qP = solve_price_single(bP, bPp, NP)
            else:
                itauP = draw_from_probs(piTau_P_def_sole[ks,:])
                ibPp = 0; qP = 0.0

            # O excluded tau on the fly (exact EV-over-tau)
            WtauO = np.full(NT, -1e18, dtype=np.float64)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    continue
                g = float(rev_defA[izO,it])
                EV_O_cont, _, *_ = exp_next_objects(izz, 1, 0, 0, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)
                WtauO[it] = float(u_defA[izO,it]) + float(v_g(g)) + beta*EV_O_cont
            itauO = draw_from_probs(softmax_tau(WtauO, rhotau).astype(np.float32))
            dO_today = 0; ibOp = 0; qO = 0.0

        else:
            # both excluded: draw taus on the fly (exact EV-over-tau)
            WtauO = np.full(NT, -1e18, dtype=np.float64)
            for it in range(NT):
                if ok_defA[izO,it] == 0:
                    continue
                g = float(rev_defA[izO,it])
                EV_O_cont, _, *_ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)
                WtauO[it] = float(u_defA[izO,it]) + float(v_g(g)) + beta*EV_O_cont
            itauO = draw_from_probs(softmax_tau(WtauO, rhotau).astype(np.float32))

            WtauP = np.full(NT, -1e18, dtype=np.float64)
            for it in range(NT):
                if ok_defA[izP,it] == 0:
                    continue
                g = float(rev_defA[izP,it])
                _, EV_P_cont, *_ = exp_next_objects(izz, 1, 1, 0, 0, 0, 0, V_O, V_P, d_O, d_P, muL, sid_of)
                WtauP[it] = float(u_defA[izP,it]) + float(v_g(g)) + beta*EV_P_cont
            itauP = draw_from_probs(softmax_tau(WtauP, rhotau).astype(np.float32))

            dO_today = 0; dP_today = 0
            ibOp = 0; ibPp = 0
            qO = 0.0; qP = 0.0

        # --- Realized allocations and budgets (exact) ---
        if mO==1 or dO_today==1:
            lO, cO, uO, revO, okO = household_block(float(z_def[izO]), float(tau_grid[itauO]))
            yO = float(z_def[izO]) * lO
            gO = revO
        else:
            lO, cO, uO, revO, okO = household_block(float(z_grid[izO]), float(tau_grid[itauO]))
            yO = float(z_grid[izO]) * lO
            gO = revO + qO*float(b_grid[ibOp]) - bO

        if mP==1 or dP_today==1:
            lP, cP, uP, revP, okP = household_block(float(z_def[izP]), float(tau_grid[itauP]))
            yP = float(z_def[izP]) * lP
            gP = revP
        else:
            lP, cP, uP, revP, okP = household_block(float(z_grid[izP]), float(tau_grid[itauP]))
            yP = float(z_grid[izP]) * lP
            gP = revP + qP*float(b_grid[ibPp]) - bP

        repayments = (1-dO_today)*bO + (1-dP_today)*bP
        purchases  = qO*float(b_grid[ibOp]) + qP*float(b_grid[ibPp])
        CL = yL + repayments - purchases
        if CL <= 1e-14:
            CL = 1e-14
        mu = CL**(-sigmaL)

        # --- Next shocks ---
        izz_next = draw_next_izz(izz)

        # --- Next regimes and debts (exact) ---
        mO_next = 1 if (mO==0 and dO_today==1) else (mO if mO==1 else 0)
        mP_next = 1 if (mP==0 and dP_today==1) else (mP if mP==1 else 0)

        if mO_next==1 and np.random.rand() < lam:
            mO_next = 0
        if mP_next==1 and np.random.rand() < lam:
            mP_next = 0

        r_next = m_to_r(mO_next, mP_next)

        ibO_next = 0 if mO_next==1 else (ibOp if (mO==0 and dO_today==0) else 0)
        ibP_next = 0 if mP_next==1 else (ibPp if (mP==0 and dP_today==0) else 0)

        sid = int(sid_of[izz_next, ibO_next, ibP_next, r_next])

        # store post burn-in
        if t >= burnin:
            tt = t - burnin

            paths["O_y"][n,tt]=yO
            paths["O_c"][n,tt]=cO
            paths["O_l"][n,tt]=lO
            paths["O_g"][n,tt]=gO
            paths["O_tau"][n,tt]=tau_grid[itauO]
            paths["O_b"][n,tt]=bO
            paths["O_q"][n,tt]=qO
            paths["O_def"][n,tt]=dO_today

            paths["P_y"][n,tt]=yP
            paths["P_c"][n,tt]=cP
            paths["P_l"][n,tt]=lP
            paths["P_g"][n,tt]=gP
            paths["P_tau"][n,tt]=tau_grid[itauP]
            paths["P_b"][n,tt]=bP
            paths["P_q"][n,tt]=qP
            paths["P_def"][n,tt]=dP_today

            paths["L_C"][n,tt]=CL
            paths["L_mu"][n,tt]=mu

print("Simulation finished.", flush=True)

# ============================================================
# Event study around O default events (exact)
# ============================================================

H = 20
win = np.arange(-H, H+1)

def event_average(series, events, H):
    acc = np.zeros(2*H+1)
    cnt = 0
    for n in range(series.shape[0]):
        ev_t = np.where(events[n])[0]
        for t0 in ev_t:
            if t0-H < 0 or t0+H >= series.shape[1]:
                continue
            acc += series[n, t0-H:t0+H+1]
            cnt += 1
    if cnt == 0:
        return np.full(2*H+1, np.nan), 0
    return acc/cnt, cnt

events_O = paths["O_def"].astype(bool)

vars_to_plot = [
    ("O_y","O output"), ("O_c","O consumption"), ("O_l","O labor"), ("O_g","O public g"), ("O_tau","O tax"), ("O_b","O debt due"), ("O_q","O bond price"),
    ("P_y","P output"), ("P_c","P consumption"), ("P_l","P labor"), ("P_g","P public g"), ("P_tau","P tax"), ("P_b","P debt due"), ("P_q","P bond price"),
    ("L_C","Lender consumption"), ("L_mu","Lender MU"),
]

print("\nGenerating event study plots...", flush=True)
fig, axes = plt.subplots(6, 3, figsize=(15, 16))
axes = axes.ravel()

for i, (k, title) in enumerate(vars_to_plot):
    avg, cnt = event_average(paths[k], events_O, H)
    ax = axes[i]
    ax.plot(win, avg)
    ax.axvline(0, linestyle="--")
    ax.set_title(f"{title} (events={cnt})")
    ax.set_xlabel("t - O default")
    ax.grid(True)

for j in range(len(vars_to_plot), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig('/workspace/event_study.png', dpi=150)
print("Event study plot saved to /workspace/event_study.png", flush=True)

# Print summary statistics
print("\n" + "="*60, flush=True)
print("SUMMARY STATISTICS", flush=True)
print("="*60, flush=True)
print(f"O default rate: {paths['O_def'].mean()*100:.2f}%", flush=True)
print(f"P default rate: {paths['P_def'].mean()*100:.2f}%", flush=True)
print(f"Mean O debt: {paths['O_b'].mean():.4f}", flush=True)
print(f"Mean P debt: {paths['P_b'].mean():.4f}", flush=True)
print(f"Mean O tax rate: {paths['O_tau'].mean():.4f}", flush=True)
print(f"Mean P tax rate: {paths['P_tau'].mean():.4f}", flush=True)
print(f"Mean O bond price: {paths['O_q'].mean():.4f}", flush=True)
print(f"Mean P bond price: {paths['P_q'].mean():.4f}", flush=True)
print(f"Mean lender consumption: {paths['L_C'].mean():.4f}", flush=True)
