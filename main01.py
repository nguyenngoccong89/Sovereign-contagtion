import numpy as np
from dataclasses import dataclass

# ============================================================
# 0. Params / grids (your calibration table + solver knobs)
# ============================================================

@dataclass
class Params:
    # Household + public goods
    sigma_H: float = 2.0
    theta: float = 1.0
    nu: float = 2.0
    omega_g: float = 0.40
    beta: float = 0.90

    # Productivity
    rho_z: float = 0.90
    sigma_eps: float = 0.02
    NZ: int = 7
    tauchen_m: float = 3.0

    # Default penalty: z_def = min(z, delta*zbar)
    delta_def: float = 0.90

    # Market access: Pr(reenter | excluded) = lambda
    lam_reentry: float = 0.20

    # Lender (CRRA)
    sigma_L: float = 2.5
    beta_L: float = 0.98
    y_L: float = 0.80

    # EV smoothing scales
    rho_a: float = 1e-3   # repay action smoothing over (b',tau)
    rho_d: float = 1e-3   # default vs repay smoothing
    euler_gamma: float = 0.5772156649015329

    # Iteration / damping
    max_iter: int = 400
    tol: float = 1e-7
    damp: float = 0.5

    # Newton for lender-consumption root (pricing)
    newton_maxit: int = 30
    newton_tol: float = 1e-12


@dataclass
class Grids:
    bO: np.ndarray
    bP: np.ndarray
    tauO: np.ndarray
    tauP: np.ndarray
    iBO0: int
    iBP0: int


def make_grids_10():
    # 10 grid points each for B and tau (your request)
    bO = np.linspace(0.0, 0.08, 10)
    bP = np.linspace(0.0, 0.08, 10)
    tauO = np.linspace(0.0, 0.30, 10)
    tauP = np.linspace(0.0, 0.30, 10)
    iBO0 = int(np.argmin(np.abs(bO - 0.0)))
    iBP0 = int(np.argmin(np.abs(bP - 0.0)))
    # enforce exact zeros if close
    bO[iBO0] = 0.0
    bP[iBP0] = 0.0
    return Grids(bO=bO, bP=bP, tauO=tauO, tauP=tauP, iBO0=iBO0, iBP0=iBP0)


# ============================================================
# 1. Tauchen + joint process (no correlation)
# ============================================================

def _norm_cdf(x):
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))

def tauchen_logz(N, rho, sigma_eps, m):
    sigma_z = sigma_eps / np.sqrt(1 - rho**2)
    zmax = m * sigma_z
    zmin = -zmax
    grid = np.linspace(zmin, zmax, N)
    step = grid[1] - grid[0]

    Pi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j == 0:
                Pi[i, j] = _norm_cdf((grid[0] - rho * grid[i] + step / 2) / sigma_eps)
            elif j == N - 1:
                Pi[i, j] = 1 - _norm_cdf((grid[-1] - rho * grid[i] - step / 2) / sigma_eps)
            else:
                z_low = (grid[j] - rho * grid[i] - step / 2) / sigma_eps
                z_high = (grid[j] - rho * grid[i] + step / 2) / sigma_eps
                Pi[i, j] = _norm_cdf(z_high) - _norm_cdf(z_low)

    # stationary dist
    pi = np.ones(N) / N
    for _ in range(20000):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < 1e-14:
            break
        pi = pi_new

    return grid, np.exp(grid), Pi, pi


def build_joint_z(params: Params):
    logz, z, Pi, pi = tauchen_logz(params.NZ, params.rho_z, params.sigma_eps, params.tauchen_m)
    # independence => kron
    Pi2 = np.kron(Pi, Pi)            # NZ^2 x NZ^2
    zO = np.repeat(z, params.NZ)     # length NZ^2
    zP = np.tile(z, params.NZ)

    zbar = float(np.sum(pi * z))
    z_def = np.minimum(z, params.delta_def * zbar)
    zO_def = np.repeat(z_def, params.NZ)
    zP_def = np.tile(z_def, params.NZ)

    return zO, zP, zO_def, zP_def, Pi2


# ============================================================
# 2. Static household block (GHH + consumption tax)
# ============================================================

def ghh_labor(z, tau, theta, nu):
    # l(z,tau) = ( z / (theta*(1+tau)) )^(1/nu)
    return (z / (theta * (1.0 + tau))) ** (1.0 / nu)

def private_utility(c, l, params: Params):
    inside = c - params.theta * (l ** (1 + params.nu)) / (1 + params.nu)
    u = np.full_like(inside, -1e12, dtype=float)
    ok = inside > 0
    u[ok] = (inside[ok] ** (1 - params.sigma_H)) / (1 - params.sigma_H)
    return u

def v_public(g, params: Params):
    g = np.asarray(g, dtype=float)
    out = np.full_like(g, -1e12, dtype=float)
    ok = g > 0
    out[ok] = params.omega_g * (g[ok] ** (1 - params.sigma_H)) / (1 - params.sigma_H)
    return out


@dataclass
class StaticBlock:
    # arrays [NZ2, NT]
    uO: np.ndarray
    TO: np.ndarray
    uO_def: np.ndarray
    TO_def: np.ndarray

    uP: np.ndarray
    TP: np.ndarray
    uP_def: np.ndarray
    TP_def: np.ndarray


def precompute_static(params: Params, grids: Grids, zO, zP, zO_def, zP_def):
    NZ2 = zO.size

    # O normal
    tauO = grids.tauO
    lO = ghh_labor(zO[:, None], tauO[None, :], params.theta, params.nu)
    yO = zO[:, None] * lO
    cO = yO / (1.0 + tauO[None, :])
    TO = tauO[None, :] * cO
    uO = private_utility(cO, lO, params)

    # O default productivity
    lO_d = ghh_labor(zO_def[:, None], tauO[None, :], params.theta, params.nu)
    yO_d = zO_def[:, None] * lO_d
    cO_d = yO_d / (1.0 + tauO[None, :])
    TO_d = tauO[None, :] * cO_d
    uO_d = private_utility(cO_d, lO_d, params)

    # P normal
    tauP = grids.tauP
    lP = ghh_labor(zP[:, None], tauP[None, :], params.theta, params.nu)
    yP = zP[:, None] * lP
    cP = yP / (1.0 + tauP[None, :])
    TP = tauP[None, :] * cP
    uP = private_utility(cP, lP, params)

    # P default productivity
    lP_d = ghh_labor(zP_def[:, None], tauP[None, :], params.theta, params.nu)
    yP_d = zP_def[:, None] * lP_d
    cP_d = yP_d / (1.0 + tauP[None, :])
    TP_d = tauP[None, :] * cP_d
    uP_d = private_utility(cP_d, lP_d, params)

    return StaticBlock(uO=uO, TO=TO, uO_def=uO_d, TO_def=TO_d,
                       uP=uP, TP=TP, uP_def=uP_d, TP_def=TP_d)


# ============================================================
# 3. EV helpers (logsumexp + default logit)
# ============================================================

def logsumexp_and_probs(W, rho):
    """
    inclusive = rho * log(sum(exp(W/rho))) and softmax probs
    W: (..., A)
    """
    W = np.asarray(W, dtype=float)
    m = np.max(W, axis=-1, keepdims=True)
    ex = np.exp((W - m) / rho)
    S = np.sum(ex, axis=-1, keepdims=True)
    probs = ex / S
    inc = rho * (np.log(S) + m / rho)
    return inc[..., 0], probs

def default_prob(Vdef, Vrep, rho_d):
    mx = np.maximum(Vdef, Vrep)
    num = np.exp((Vdef - mx) / rho_d)
    den = num + np.exp((Vrep - mx) / rho_d)
    return num / den

def inclusive_two(Vdef, Vrep, rho_d, gamma):
    mx = np.maximum(Vdef, Vrep)
    inc = gamma * rho_d + rho_d * (np.log(np.exp((Vdef - mx) / rho_d) + np.exp((Vrep - mx) / rho_d)) + mx / rho_d)
    return inc


# ============================================================
# 4. Price solver for one-period bonds (Appendix-C style)
#    For a given branch, solve lender consumption c then q = N * c^{sigma_L}
#    where N = beta_L E[ mu_{t+1} (1-d_{t+1}) ].
# ============================================================

def solve_c_newton(A, K, sigma_L, maxit=30, tol=1e-12):
    """
    Solve scalar equation: c + c^{sigma_L} * K = A
    Vectorized over c if A,K are arrays.
    """
    c = np.maximum(A, 1e-12).astype(float)
    for _ in range(maxit):
        c_pow = c ** sigma_L
        f = c + c_pow * K - A
        if np.max(np.abs(f)) < tol:
            break
        df = 1.0 + sigma_L * (c ** (sigma_L - 1.0)) * K
        c_new = c - f / df
        c = np.maximum(c_new, 1e-12)
    return c


# ============================================================
# 5. Container for equilibrium objects (4 regimes)
# ============================================================

@dataclass
class Eq:
    # Values V_O, V_P over full state (z2, bO, bP, mO, mP)
    VO: np.ndarray
    VP: np.ndarray

    # Default probabilities in market states (set =1 in excluded)
    dO: np.ndarray
    dP: np.ndarray

    # Lender marginal utility mu = C_L^{-sigma_L}
    muL: np.ndarray

    # Expected repay policies conditional on repay, and tau in autarky/default
    EbO_rep: np.ndarray
    EtauO_rep: np.ndarray
    EtauO_def: np.ndarray

    EbP_rep: np.ndarray
    EtauP_rep: np.ndarray
    EtauP_def: np.ndarray

    # Conditional P objects for simulation in regime (0,0):
    dP_if_Orep: np.ndarray
    EbP_if_Orep: np.ndarray
    EtauP_if_Orep: np.ndarray
    dP_if_Odef: np.ndarray
    EbP_if_Odef: np.ndarray
    EtauP_if_Odef: np.ndarray

    # Prices for regime-specific branches (for simulation / diagnostics)
    # (0,0): policy-averaged prices (not profile)
    qO_pol: np.ndarray
    qP_pol: np.ndarray


def init_eq(params: Params, grids: Grids):
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    shape = (NZ2, NB, NB, 2, 2)

    VO = np.zeros(shape)
    VP = np.zeros(shape)

    dO = np.zeros(shape)
    dP = np.zeros(shape)

    muL = np.full(shape, params.y_L ** (-params.sigma_L))

    EbO_rep = np.zeros(shape); EtauO_rep = np.zeros(shape); EtauO_def = np.zeros(shape)
    EbP_rep = np.zeros(shape); EtauP_rep = np.zeros(shape); EtauP_def = np.zeros(shape)

    dP_if_Orep = np.zeros(shape); EbP_if_Orep = np.zeros(shape); EtauP_if_Orep = np.zeros(shape)
    dP_if_Odef = np.zeros(shape); EbP_if_Odef = np.zeros(shape); EtauP_if_Odef = np.zeros(shape)

    qO_pol = np.zeros(shape); qP_pol = np.zeros(shape)

    # excluded states: force default prob=1 (pricing payoff 0 if accidentally referenced)
    dO[:, :, :, 1, :] = 1.0
    dP[:, :, :, :, 1] = 1.0

    return Eq(VO, VP, dO, dP, muL,
              EbO_rep, EtauO_rep, EtauO_def,
              EbP_rep, EtauP_rep, EtauP_def,
              dP_if_Orep, EbP_if_Orep, EtauP_if_Orep,
              dP_if_Odef, EbP_if_Odef, EtauP_if_Odef,
              qO_pol, qP_pol)


# ============================================================
# 6. Expectations over z' for continuation and for numerators
# ============================================================

def expected_over_z(Pi2, X):
    """
    For each current z2, compute E_{z'}[X(z')].
    If X has shape (NZ2, ...), returns shape (NZ2, ...).
    """
    NZ2 = Pi2.shape[0]
    Xr = X.reshape(NZ2, -1)
    EX = Pi2 @ Xr
    return EX.reshape(X.shape)


def build_expectations(params: Params, grids: Grids, Pi2, eq: Eq):
    """
    Build:
      EV_VO[mO,mP]: E_z'[ VO(z', bO, bP, mO,mP) ] for all bO,bP
      EV_VP similarly
    and:
      N_O[mO,mP] = beta_L E_z'[ mu(z',bO,bP,mO,mP) * (1-dO(z',...)) ]
      N_P similarly
    """
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size

    EV_VO = np.zeros((2, 2, NZ2, NB, NB))
    EV_VP = np.zeros((2, 2, NZ2, NB, NB))
    N_O = np.zeros((2, 2, NZ2, NB, NB))
    N_P = np.zeros((2, 2, NZ2, NB, NB))

    # slices by next-market states
    for mO in (0, 1):
        for mP in (0, 1):
            VO_slice = eq.VO[:, :, :, mO, mP]
            VP_slice = eq.VP[:, :, :, mO, mP]
            EV_VO[mO, mP] = expected_over_z(Pi2, VO_slice)
            EV_VP[mO, mP] = expected_over_z(Pi2, VP_slice)

            X_O = eq.muL[:, :, :, mO, mP] * (1.0 - eq.dO[:, :, :, mO, mP])
            X_P = eq.muL[:, :, :, mO, mP] * (1.0 - eq.dP[:, :, :, mO, mP])
            N_O[mO, mP] = params.beta_L * expected_over_z(Pi2, X_O)
            N_P[mO, mP] = params.beta_L * expected_over_z(Pi2, X_P)

    return EV_VO, EV_VP, N_O, N_P


# ============================================================
# 7. Core regime solver: (mO,mP) = (0,0)
#    Fully Appendix-C: Stackelberg + four default outcomes + three price schedules
# ============================================================

def solve_regime_00(params: Params, grids: Grids, static: StaticBlock, EV_VO, EV_VP, N_O, N_P):
    """
    Returns dictionaries of updated arrays for regime (0,0) only:
      VO00[z2,iBO,iBP], VP00[...], dO00[...], dP00[...], mu00[...]
      plus conditional P objects for simulation and policy-averaged prices.
    """
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    NT_O = grids.tauO.size
    NT_P = grids.tauP.size

    bOgrid = grids.bO
    bPgrid = grids.bP

    iBO0 = grids.iBO0
    iBP0 = grids.iBP0

    # outputs
    VO00 = np.zeros((NZ2, NB, NB))
    VP00 = np.zeros((NZ2, NB, NB))
    dO00 = np.zeros((NZ2, NB, NB))
    dP00 = np.zeros((NZ2, NB, NB))
    mu00 = np.zeros((NZ2, NB, NB))

    # expected repay policies (conditional on repay)
    EbO_rep = np.zeros((NZ2, NB, NB))
    EtauO_rep = np.zeros((NZ2, NB, NB))
    EtauO_def = np.zeros((NZ2, NB, NB))

    EbP_rep = np.zeros((NZ2, NB, NB))
    EtauP_rep = np.zeros((NZ2, NB, NB))
    EtauP_def = np.zeros((NZ2, NB, NB))

    # conditional P objects for simulation
    dP_if_Orep = np.zeros((NZ2, NB, NB))
    EbP_if_Orep = np.zeros((NZ2, NB, NB))
    EtauP_if_Orep = np.zeros((NZ2, NB, NB))
    dP_if_Odef = np.zeros((NZ2, NB, NB))
    EbP_if_Odef = np.zeros((NZ2, NB, NB))
    EtauP_if_Odef = np.zeros((NZ2, NB, NB))

    # policy-averaged prices
    qO_pol = np.zeros((NZ2, NB, NB))
    qP_pol = np.zeros((NZ2, NB, NB))

    # --- price numerators for relevant NEXT market states ---
    # both repay => next (0,0)
    NO_00 = N_O[0, 0]  # [z2, bO', bP']
    NP_00 = N_P[0, 0]

    # only O issues when P defaults => next (0,1) with bP'=0
    NO_01 = N_O[0, 1]  # [z2, bO', bP']
    # only P issues when O defaults => next (1,0) with bO'=0
    NP_10 = N_P[1, 0]

    # --- continuation expectations for relevant NEXT states ---
    EVOO_00 = EV_VO[0, 0]
    EVOO_01 = EV_VO[0, 1]
    EVOO_10 = EV_VO[1, 0]
    EVOO_11 = EV_VO[1, 1]

    EVPP_00 = EV_VP[0, 0]
    EVPP_01 = EV_VP[0, 1]
    EVPP_10 = EV_VP[1, 0]
    EVPP_11 = EV_VP[1, 1]

    # For speed, build P action index grids once (bP',tauP):
    # WrepP[bP', tauP] but we will compute as (tauP x bP') = (NT_P x NB)
    tauP = grids.tauP
    tauO = grids.tauO

    for iz2 in range(NZ2):
        # Pre-slice static arrays at this z2
        uO_tau = static.uO[iz2, :]       # (NT_O,)
        TO_tau = static.TO[iz2, :]       # (NT_O,)
        uO_d_tau = static.uO_def[iz2, :]
        TO_d_tau = static.TO_def[iz2, :]

        uP_tau = static.uP[iz2, :]       # (NT_P,)
        TP_tau = static.TP[iz2, :]
        uP_d_tau = static.uP_def[iz2, :]
        TP_d_tau = static.TP_def[iz2, :]

        # price numerators at this z2
        NO00 = NO_00[iz2, :, :]  # (NB x NB) over (bO',bP')
        NP00 = NP_00[iz2, :, :]
        NO01_bO = NO_01[iz2, :, iBP0]  # (NB,) only O issues, bP'=0
        NP10_bP = NP_10[iz2, iBO0, :]  # (NB,) only P issues, bO'=0

        # continuation at this z2
        EVO00 = EVOO_00[iz2, :, :]   # (NB x NB)
        EVO01 = EVOO_01[iz2, :, :]   # (NB x NB) but we'll use bP=0 column
        EVO10 = EVOO_10[iz2, :, :]   # use bO=0 row for O default branch
        EVO11 = EVOO_11[iz2, :, :]

        EVP00 = EVPP_00[iz2, :, :]
        EVP01 = EVPP_01[iz2, :, :]
        EVP10 = EVPP_10[iz2, :, :]
        EVP11 = EVPP_11[iz2, :, :]

        for iBO in range(NB):
            bO_now = bOgrid[iBO]
            for iBP in range(NB):
                bP_now = bPgrid[iBP]

                # ------------------------------------------------------------
                # A) PRICE SCHEDULES for this (z2, bO, bP) in regime (0,0)
                # ------------------------------------------------------------
                # 1) both issue (none default): solve c matrix for each (bO',bP')
                A_both = params.y_L + bO_now + bP_now
                K_both = NO00 * bOgrid[:, None] + NP00 * bPgrid[None, :]
                c_both = solve_c_newton(A_both, K_both, params.sigma_L,
                                        maxit=params.newton_maxit, tol=params.newton_tol)
                c_pow = c_both ** params.sigma_L
                qO_both = NO00 * c_pow
                qP_both = NP00 * c_pow

                # 2) only O issues (P defaults): c vector over bO'
                A_onlyO = params.y_L + bO_now
                K_onlyO = NO01_bO * bOgrid
                c_onlyO = solve_c_newton(A_onlyO, K_onlyO, params.sigma_L,
                                         maxit=params.newton_maxit, tol=params.newton_tol)
                qO_onlyO = NO01_bO * (c_onlyO ** params.sigma_L)

                # 3) only P issues (O defaults): c vector over bP'
                A_onlyP = params.y_L + bP_now
                K_onlyP = NP10_bP * bPgrid
                c_onlyP = solve_c_newton(A_onlyP, K_onlyP, params.sigma_L,
                                         maxit=params.newton_maxit, tol=params.newton_tol)
                qP_onlyP = NP10_bP * (c_onlyP ** params.sigma_L)

                # ------------------------------------------------------------
                # B) FOLLOWER (P) given O REPAYS with a chosen bO'
                #    Note: P's response depends on O's bO' (NOT on tauO).
                # ------------------------------------------------------------
                # For each O debt choice index jBOp, compute:
                #   - P repay inclusive value VrepP[jBOp]
                #   - P default value VdefP[jBOp]
                #   - dP_given_Orep[jBOp]
                #   - p_bP_condrep[jBOp, kBPp]
                #   - expected (bP', tauP) conditional on repay
                VP_cond_Orep = np.zeros(NB)
                dP_cond_Orep = np.zeros(NB)
                p_bP_condrep = np.zeros((NB, NB))
                EbP_condrep = np.zeros(NB)
                EtauP_condrep = np.zeros(NB)
                EtauP_def_cond = np.zeros(NB)

                for jBOp in range(NB):
                    # ----- P repay value index over (tauP,bP') using qP_both[jBOp, kBPp]
                    # Build Wrep as (NT_P x NB)
                    qP_row = qP_both[jBOp, :]   # depends on bP'
                    # gP(tau,bP') = TP(tau) + qP(bP')*bP' - bP_now
                    gP = TP_tau[:, None] + qP_row[None, :] * bPgrid[None, :] - bP_now
                    Urep = uP_tau[:, None] + v_public(gP, params)
                    # continuation if repay: next state (0,0), debts (bO',bP')
                    Cont = params.beta * EVP00[jBOp, :][None, :]
                    Wrep = Urep + Cont

                    Vrep_inc, probs = logsumexp_and_probs(Wrep.reshape(-1), params.rho_a)
                    VrepP = params.euler_gamma * params.rho_a + Vrep_inc

                    # conditional-on-repay expectations
                    probs2 = probs.reshape(NT_P, NB)
                    p_bP = probs2.sum(axis=0)                    # marginal over bP'
                    EbP = float(np.sum(p_bP * bPgrid))
                    Etau = float(np.sum(probs2.sum(axis=1) * tauP))

                    # ----- P default value: choose tauP in default/autarky, next state (0,1), debt (bO',0)
                    Cont_def = params.beta * EVP01[jBOp, iBP0]
                    Wdef_tau = uP_d_tau + v_public(TP_d_tau, params) + Cont_def
                    Vdef_inc, p_tau_def = logsumexp_and_probs(Wdef_tau, params.rho_a)
                    VdefP = params.euler_gamma * params.rho_a + Vdef_inc
                    Etau_def = float(np.sum(p_tau_def * tauP))

                    dP_here = float(default_prob(VdefP, VrepP, params.rho_d))
                    VP_here = float(inclusive_two(VdefP, VrepP, params.rho_d, params.euler_gamma))

                    VP_cond_Orep[jBOp] = VP_here
                    dP_cond_Orep[jBOp] = dP_here
                    p_bP_condrep[jBOp, :] = p_bP
                    EbP_condrep[jBOp] = EbP
                    EtauP_condrep[jBOp] = Etau
                    EtauP_def_cond[jBOp] = Etau_def

                # ------------------------------------------------------------
                # C) FOLLOWER (P) when O DEFAULTS (only P issues)
                # ------------------------------------------------------------
                # P repay with qP_onlyP, continuation next (1,0); P default next (1,1).
                # Here O's next debt is 0 (iBO0).
                qP = qP_onlyP  # (NB,)
                gP = TP_tau[:, None] + qP[None, :] * bPgrid[None, :] - bP_now
                Urep = uP_tau[:, None] + v_public(gP, params)
                Cont = params.beta * EVP10[iBO0, :][None, :]
                Wrep = Urep + Cont
                Vrep_inc, probs = logsumexp_and_probs(Wrep.reshape(-1), params.rho_a)
                VrepP_od = params.euler_gamma * params.rho_a + Vrep_inc
                probs2 = probs.reshape(NT_P, NB)
                p_bP_od = probs2.sum(axis=0)
                EbP_od = float(np.sum(p_bP_od * bPgrid))
                EtauP_od = float(np.sum(probs2.sum(axis=1) * tauP))

                Cont_def = params.beta * EVP11[iBO0, iBP0]
                Wdef_tau = uP_d_tau + v_public(TP_d_tau, params) + Cont_def
                Vdef_inc, p_tau_def = logsumexp_and_probs(Wdef_tau, params.rho_a)
                VdefP_od = params.euler_gamma * params.rho_a + Vdef_inc
                dP_od = float(default_prob(VdefP_od, VrepP_od, params.rho_d))
                VP_od = float(inclusive_two(VdefP_od, VrepP_od, params.rho_d, params.euler_gamma))
                EtauP_def_od = float(np.sum(p_tau_def * tauP))

                # ------------------------------------------------------------
                # D) LEADER (O) repay value over (bO',tauO) anticipating P
                # ------------------------------------------------------------
                # For each jBOp and tauO, compute expected value:
                #   With prob (1-dP_cond_Orep[jBOp]): P repays and chooses bP' ~ p_bP_condrep[jBOp]
                #   With prob dP_cond_Orep[jBOp]: P defaults (O issues alone at qO_onlyO[jBOp])
                #
                # Use expectation over bP' because qO_both and continuation depend on bP'.
                W_O_rep = np.full((NB, NT_O), -1e12, dtype=float)

                for jBOp in range(NB):
                    bO_prime = bOgrid[jBOp]
                    dPj = dP_cond_Orep[jBOp]
                    p_bPj = p_bP_condrep[jBOp, :]  # conditional on P repaying

                    # Part when P repays: expectation over bP' only (tauP integrated out)
                    # Flow: uO(tauO) + E_{bP'}[ v( TO(tauO)+qO_both(jBOp,bP')*bO'-bO_now ) ]
                    qO_row = qO_both[jBOp, :]  # depends on bP'
                    g_rep = TO_tau[:, None] + (qO_row[None, :] * bO_prime) - bO_now
                    flow_rep = uO_tau[:, None] + v_public(g_rep, params)

                    # continuation: beta * E_{bP'}[ EVO00(jBOp,bP') ]
                    cont_rep = params.beta * EVO00[jBOp, :][None, :]

                    # expectation over bP' using p_bPj
                    EV_profile = (flow_rep + cont_rep) @ p_bPj

                    # Part when P defaults: O issues alone (qO_onlyO[jBOp]) and continuation to (0,1) with bP=0
                    g_def = TO_tau + qO_onlyO[jBOp] * bO_prime - bO_now
                    flow_def = uO_tau + v_public(g_def, params)
                    cont_def_O = params.beta * EVO01[jBOp, iBP0]
                    EV_defbranch = flow_def + cont_def_O

                    # Total W for each tauO
                    W_O_rep[jBOp, :] = (1.0 - dPj) * EV_profile + dPj * EV_defbranch

                # inclusive over O repay actions
                Vrep_inc, probs_O = logsumexp_and_probs(W_O_rep.reshape(-1), params.rho_a)
                VrepO = params.euler_gamma * params.rho_a + Vrep_inc
                probsO2 = probs_O.reshape(NB, NT_O)
                p_bO = probsO2.sum(axis=1)  # marginal over bO'

                EbO = float(np.sum(p_bO * bOgrid))
                EtO = float(np.sum(probsO2.sum(axis=0) * tauO))

                # ------------------------------------------------------------
                # E) O default value: choose tauO in default and anticipate P under O default
                # ------------------------------------------------------------
                # O default flow uses (uO_def + v(TO_def)), no repayment of bO_now, no issuance
                # continuation uses P outcomes under O default:
                #   if P repays: next (1,0), debts (0,bP')
                #   if P defaults: next (1,1), debts (0,0)
                cont_Odef = params.beta * (
                    (1.0 - dP_od) * np.sum(p_bP_od * EVO10[iBO0, :]) +
                    dP_od * EVO11[iBO0, iBP0]
                )
                W_Odef_tau = uO_d_tau + v_public(TO_d_tau, params) + cont_Odef
                Vdef_inc, p_tau_defO = logsumexp_and_probs(W_Odef_tau, params.rho_a)
                VdefO = params.euler_gamma * params.rho_a + Vdef_inc
                EtO_def = float(np.sum(p_tau_defO * tauO))

                # default probability for O
                dO_here = float(default_prob(VdefO, VrepO, params.rho_d))
                VO_here = float(inclusive_two(VdefO, VrepO, params.rho_d, params.euler_gamma))

                # ------------------------------------------------------------
                # F) Aggregate P value & default prob given O mixes and defaults
                # ------------------------------------------------------------
                # If O repays: O chooses bO' distribution p_bO, then P gets VP_cond_Orep[bO']
                VP_if_Orep = float(np.sum(p_bO * VP_cond_Orep))
                dP_if_Orep_here = float(np.sum(p_bO * dP_cond_Orep))
                EbP_if_Orep_here = float(np.sum(p_bO * EbP_condrep))
                EtP_if_Orep_here = float(np.sum(p_bO * EtauP_condrep))

                # If O defaults: VP_od
                VP_here = (1.0 - dO_here) * VP_if_Orep + dO_here * VP_od

                dP_here = (1.0 - dO_here) * dP_if_Orep_here + dO_here * dP_od

                # ------------------------------------------------------------
                # G) Lender expected consumption and mu in regime (0,0)
                # ------------------------------------------------------------
                # Branch O repays:
                # For each bO' with prob p_bO:
                #   - P repays w.p. (1-dP_cond_Orep[bO'])
                #       lender C = y_L + bO_now + bP_now - E_{bP'|rep}[ qO_both*bO' + qP_both*bP' ]
                #   - P defaults w.p. dP_cond_Orep[bO']
                #       lender C = y_L + bO_now - qO_onlyO*bO'
                #
                # Branch O defaults:
                #   - P repays w.p. (1-dP_od):
                #       lender C = y_L + bP_now - E_{bP'|rep}[ qP_onlyP*bP' ]
                #   - P defaults w.p. dP_od:
                #       lender C = y_L
                #
                # Compute expectations:
                C_Orep = 0.0
                qO_pol_sum = 0.0
                qP_pol_sum = 0.0
                EbP_rep_sum = 0.0
                EtP_rep_sum = 0.0
                EtP_def_sum = 0.0

                for jBOp in range(NB):
                    pb = p_bO[jBOp]
                    if pb == 0:
                        continue
                    bO_prime = bOgrid[jBOp]
                    dPj = dP_cond_Orep[jBOp]
                    p_bPj = p_bP_condrep[jBOp, :]

                    # if P repays
                    # expected purchase cost: E[qO_both*bO' + qP_both*bP']
                    cost_both = float(np.sum(p_bPj * (qO_both[jBOp, :] * bO_prime + qP_both[jBOp, :] * bPgrid)))
                    C_both = params.y_L + bO_now + bP_now - cost_both

                    # if P defaults
                    C_onlyO = params.y_L + bO_now - qO_onlyO[jBOp] * bO_prime

                    C_Orep += pb * ((1.0 - dPj) * C_both + dPj * C_onlyO)

                    # policy-averaged prices for reporting: expected qO, qP conditional on repay branch
                    # use P's repay distribution
                    qO_pol_sum += pb * (1.0 - dPj) * float(np.sum(p_bPj * qO_both[jBOp, :]))
                    qP_pol_sum += pb * (1.0 - dPj) * float(np.sum(p_bPj * qP_both[jBOp, :]))

                    EbP_rep_sum += pb * EbP_condrep[jBOp]
                    EtP_rep_sum += pb * EtauP_condrep[jBOp]
                    EtP_def_sum += pb * EtauP_def_cond[jBOp]

                # O default branch:
                cost_onlyP = float(np.sum(p_bP_od * (qP_onlyP * bPgrid)))
                C_onlyP = params.y_L + bP_now - cost_onlyP
                C_Odef = (1.0 - dP_od) * C_onlyP + dP_od * params.y_L

                C_total = (1.0 - dO_here) * C_Orep + dO_here * C_Odef
                C_total = max(C_total, 1e-12)
                mu_here = C_total ** (-params.sigma_L)

                # store
                VO00[iz2, iBO, iBP] = VO_here
                VP00[iz2, iBO, iBP] = VP_here
                dO00[iz2, iBO, iBP] = dO_here
                dP00[iz2, iBO, iBP] = dP_here
                mu00[iz2, iBO, iBP] = mu_here

                EbO_rep[iz2, iBO, iBP] = EbO
                EtauO_rep[iz2, iBO, iBP] = EtO
                EtauO_def[iz2, iBO, iBP] = EtO_def

                EbP_rep[iz2, iBO, iBP] = EbP_rep_sum
                EtauP_rep[iz2, iBO, iBP] = EtP_rep_sum
                EtauP_def[iz2, iBO, iBP] = EtP_def_sum

                dP_if_Orep[iz2, iBO, iBP] = dP_if_Orep_here
                EbP_if_Orep[iz2, iBO, iBP] = EbP_if_Orep_here
                EtauP_if_Orep[iz2, iBO, iBP] = EtP_if_Orep_here

                dP_if_Odef[iz2, iBO, iBP] = dP_od
                EbP_if_Odef[iz2, iBO, iBP] = EbP_od
                EtauP_if_Odef[iz2, iBO, iBP] = EtauP_od

                qO_pol[iz2, iBO, iBP] = qO_pol_sum
                qP_pol[iz2, iBO, iBP] = qP_pol_sum

    return dict(
        VO=VO00, VP=VP00, dO=dO00, dP=dP00, mu=mu00,
        EbO_rep=EbO_rep, EtauO_rep=EtauO_rep, EtauO_def=EtauO_def,
        EbP_rep=EbP_rep, EtauP_rep=EtauP_rep, EtauP_def=EtauP_def,
        dP_if_Orep=dP_if_Orep, EbP_if_Orep=EbP_if_Orep, EtauP_if_Orep=EtauP_if_Orep,
        dP_if_Odef=dP_if_Odef, EbP_if_Odef=EbP_if_Odef, EtauP_if_Odef=EtauP_if_Odef,
        qO_pol=qO_pol, qP_pol=qP_pol
    )


# ============================================================
# 8. Regimes (0,1), (1,0), (1,1)
#    These are simpler: single-issuer Stackelberg degenerates.
# ============================================================

def solve_regime_01(params: Params, grids: Grids, static: StaticBlock, EV_VO, EV_VP, N_O, N_P):
    """
    (mO,mP) = (0,1): O has access, P excluded.
    P cannot issue; P's next access uses lambda.
    O chooses default vs repay and (bO',tauO) if repay.
    """
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    NT_O = grids.tauO.size
    NT_P = grids.tauP.size

    bOgrid = grids.bO
    iBO0 = grids.iBO0
    iBP0 = grids.iBP0
    tauO = grids.tauO
    tauP = grids.tauP

    VO01 = np.zeros((NZ2, NB, NB))
    VP01 = np.zeros((NZ2, NB, NB))
    dO01 = np.zeros((NZ2, NB, NB))
    dP01 = np.ones((NZ2, NB, NB))  # excluded (no market) => treat as 1
    mu01 = np.zeros((NZ2, NB, NB))

    EbO_rep = np.zeros((NZ2, NB, NB))
    EtauO_rep = np.zeros((NZ2, NB, NB))
    EtauO_def = np.zeros((NZ2, NB, NB))

    # P excluded: only tau choice in autarky
    EtauP_aut = np.zeros((NZ2, NB, NB))
    qO_pol = np.zeros((NZ2, NB, NB))
    qP_pol = np.zeros((NZ2, NB, NB))

    # Numerators for O bond when P excluded:
    # If O repays, next mO'=0.
    # For P, mP' is 0 w.p lambda else 1, and bP' = 0 always.
    NO00 = N_O[0, 0]   # next (0,0)
    NO01 = N_O[0, 1]   # next (0,1)
    EVO00 = EV_VO[0, 0]; EVO01 = EV_VO[0, 1]; EVO10 = EV_VO[1, 0]; EVO11 = EV_VO[1, 1]
    EVP00 = EV_VP[0, 0]; EVP01 = EV_VP[0, 1]; EVP10 = EV_VP[1, 0]; EVP11 = EV_VP[1, 1]

    for iz2 in range(NZ2):
        uO_tau = static.uO[iz2, :]
        TO_tau = static.TO[iz2, :]
        uO_d_tau = static.uO_def[iz2, :]
        TO_d_tau = static.TO_def[iz2, :]

        uP_d_tau = static.uP_def[iz2, :]
        TP_d_tau = static.TP_def[iz2, :]

        # Precompute P excluded value components (depends on O policy through next mO)
        for iBO in range(NB):
            bO_now = bOgrid[iBO]
            for iBP in range(NB):
                # in excluded state we treat bP_now irrelevant; debt effectively 0
                # -------- Price schedule for O issuance in this regime ----------
                # Numerator mixture over next mP':
                NO_mix = params.lam_reentry * NO00[iz2, :, iBP0] + (1.0 - params.lam_reentry) * NO01[iz2, :, iBP0]
                A = params.y_L + bO_now
                K = NO_mix * bOgrid
                c = solve_c_newton(A, K, params.sigma_L, params.newton_maxit, params.newton_tol)
                qO = NO_mix * (c ** params.sigma_L)

                # -------- O repay value over (bO',tauO) ----------
                # continuation if O repays: next mO'=0, bO=bO'
                # next mP' mixture due to reentry:
                cont_rep = params.beta * (
                    params.lam_reentry * EVO00[iz2, :, iBP0] + (1.0 - params.lam_reentry) * EVO01[iz2, :, iBP0]
                )  # shape (NB,)

                # Wrep[jBOp, tauO]
                g = TO_tau[None, :] + (qO[:, None] * bOgrid[:, None]) - bO_now
                Wrep = uO_tau[None, :] + v_public(g, params) + cont_rep[:, None]
                Vrep_inc, pO = logsumexp_and_probs(Wrep.reshape(-1), params.rho_a)
                VrepO = params.euler_gamma * params.rho_a + Vrep_inc
                pO2 = pO.reshape(NB, NT_O)
                p_bO = pO2.sum(axis=1)
                EbO = float(np.sum(p_bO * bOgrid))
                EtO = float(np.sum(pO2.sum(axis=0) * tauO))

                # -------- O default value (next mO'=1) ----------
                # P remains excluded today; next mP' still follows reentry.
                cont_def = params.beta * (
                    params.lam_reentry * EVO10[iz2, iBO0, iBP0] + (1.0 - params.lam_reentry) * EVO11[iz2, iBO0, iBP0]
                )
                Wdef_tau = uO_d_tau + v_public(TO_d_tau, params) + cont_def
                Vdef_inc, p_tau_def = logsumexp_and_probs(Wdef_tau, params.rho_a)
                VdefO = params.euler_gamma * params.rho_a + Vdef_inc
                EtO_def = float(np.sum(p_tau_def * tauO))

                dO_here = float(default_prob(VdefO, VrepO, params.rho_d))
                VO_here = float(inclusive_two(VdefO, VrepO, params.rho_d, params.euler_gamma))

                # -------- P excluded value (chooses tau each period) ----------
                # P today excluded => chooses tauP using default productivity objects (no borrowing),
                # continuation mixes:
                #   mP' = 0 w.p lambda (reenter), else 1
                # O's next m depends on O default probability today:
                #   if O repays (prob 1-dO): mO' = 0
                #   if O defaults (prob dO): mO' = 1
                # When P reenters (mP'=0), next state is either (0,0) or (1,0) with debt bO' determined by O.
                # We approximate using O's expected repay debt EbO (smooth policy) mapped to grid.
                iBO_star = int(np.argmin(np.abs(bOgrid - EbO)))

                cont_P_reenter = (1.0 - dO_here) * EVP00[iz2, iBO_star, iBP0] + dO_here * EVP10[iz2, iBO0, iBP0]
                cont_P_stay = (1.0 - dO_here) * EVP01[iz2, iBO_star, iBP0] + dO_here * EVP11[iz2, iBO0, iBP0]
                contP = params.beta * (params.lam_reentry * cont_P_reenter + (1.0 - params.lam_reentry) * cont_P_stay)

                W_tauP = uP_d_tau + v_public(TP_d_tau, params) + contP
                VP_inc, p_tauP = logsumexp_and_probs(W_tauP, params.rho_a)
                VP_here = params.euler_gamma * params.rho_a + VP_inc
                EtP = float(np.sum(p_tauP * tauP))

                # -------- lender consumption / mu (only O can issue if it repays) ----------
                # If O repays: C = y_L + bO_now - E[qO*bO']
                # Use expected bO' under repay p_bO:
                cost = float(np.sum(p_bO * (qO * bOgrid)))
                C_rep = params.y_L + bO_now - cost
                C_def = params.y_L
                C = (1.0 - dO_here) * C_rep + dO_here * C_def
                C = max(C, 1e-12)
                mu_here = C ** (-params.sigma_L)

                # store
                VO01[iz2, iBO, iBP] = VO_here
                VP01[iz2, iBO, iBP] = VP_here
                dO01[iz2, iBO, iBP] = dO_here
                mu01[iz2, iBO, iBP] = mu_here
                EbO_rep[iz2, iBO, iBP] = EbO
                EtauO_rep[iz2, iBO, iBP] = EtO
                EtauO_def[iz2, iBO, iBP] = EtO_def
                EtauP_aut[iz2, iBO, iBP] = EtP
                qO_pol[iz2, iBO, iBP] = float(np.sum(p_bO * qO))
                qP_pol[iz2, iBO, iBP] = 0.0

    return dict(VO=VO01, VP=VP01, dO=dO01, dP=dP01, mu=mu01,
                EbO_rep=EbO_rep, EtauO_rep=EtauO_rep, EtauO_def=EtauO_def,
                EtauP_aut=EtauP_aut, qO_pol=qO_pol, qP_pol=qP_pol)


def solve_regime_10(params: Params, grids: Grids, static: StaticBlock, EV_VO, EV_VP, N_O, N_P):
    """
    (mO,mP) = (1,0): O excluded, P has access. Symmetric to regime_01.
    """
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    NT_P = grids.tauP.size
    NT_O = grids.tauO.size

    bPgrid = grids.bP
    iBO0 = grids.iBO0
    iBP0 = grids.iBP0
    tauP = grids.tauP
    tauO = grids.tauO

    VO10 = np.zeros((NZ2, NB, NB))
    VP10 = np.zeros((NZ2, NB, NB))
    dO10 = np.ones((NZ2, NB, NB))  # excluded
    dP10 = np.zeros((NZ2, NB, NB))
    mu10 = np.zeros((NZ2, NB, NB))

    EbP_rep = np.zeros((NZ2, NB, NB))
    EtauP_rep = np.zeros((NZ2, NB, NB))
    EtauP_def = np.zeros((NZ2, NB, NB))
    EtauO_aut = np.zeros((NZ2, NB, NB))
    qO_pol = np.zeros((NZ2, NB, NB))
    qP_pol = np.zeros((NZ2, NB, NB))

    # Numerators for P bond with O excluded:
    NP00 = N_P[0, 0]
    NP10 = N_P[1, 0]
    EVO00 = EV_VO[0, 0]; EVO01 = EV_VO[0, 1]; EVO10 = EV_VO[1, 0]; EVO11 = EV_VO[1, 1]
    EVP00 = EV_VP[0, 0]; EVP01 = EV_VP[0, 1]; EVP10 = EV_VP[1, 0]; EVP11 = EV_VP[1, 1]

    for iz2 in range(NZ2):
        uP_tau = static.uP[iz2, :]
        TP_tau = static.TP[iz2, :]
        uP_d_tau = static.uP_def[iz2, :]
        TP_d_tau = static.TP_def[iz2, :]

        uO_d_tau = static.uO_def[iz2, :]
        TO_d_tau = static.TO_def[iz2, :]

        for iBO in range(NB):
            for iBP in range(NB):
                bP_now = bPgrid[iBP]

                # P price schedule: numerator mixture over next mO' due to O reentry
                # next mP'=0 if P repays
                NP_mix = params.lam_reentry * NP00[iz2, iBO0, :] + (1.0 - params.lam_reentry) * NP10[iz2, iBO0, :]
                A = params.y_L + bP_now
                K = NP_mix * bPgrid
                c = solve_c_newton(A, K, params.sigma_L, params.newton_maxit, params.newton_tol)
                qP = NP_mix * (c ** params.sigma_L)

                # P repay value over (bP',tauP)
                # continuation mixes next mO' from reentry:
                cont_rep = params.beta * (
                    params.lam_reentry * EVP00[iz2, iBO0, :] + (1.0 - params.lam_reentry) * EVP10[iz2, iBO0, :]
                )  # (NB,)
                g = TP_tau[:, None] + (qP[None, :] * bPgrid[None, :]) - bP_now
                Wrep = uP_tau[:, None] + v_public(g, params) + cont_rep[None, :]
                Vrep_inc, probs = logsumexp_and_probs(Wrep.reshape(-1), params.rho_a)
                VrepP = params.euler_gamma * params.rho_a + Vrep_inc
                probs2 = probs.reshape(NT_P, NB)
                p_bP = probs2.sum(axis=0)
                EbP = float(np.sum(p_bP * bPgrid))
                EtP = float(np.sum(probs2.sum(axis=1) * tauP))

                # P default value (next mP'=1)
                cont_def = params.beta * (
                    params.lam_reentry * EVP01[iz2, iBO0, iBP0] + (1.0 - params.lam_reentry) * EVP11[iz2, iBO0, iBP0]
                )
                Wdef_tau = uP_d_tau + v_public(TP_d_tau, params) + cont_def
                Vdef_inc, p_tau_def = logsumexp_and_probs(Wdef_tau, params.rho_a)
                VdefP = params.euler_gamma * params.rho_a + Vdef_inc
                EtP_def = float(np.sum(p_tau_def * tauP))

                dP_here = float(default_prob(VdefP, VrepP, params.rho_d))
                VP_here = float(inclusive_two(VdefP, VrepP, params.rho_d, params.euler_gamma))

                # O excluded value (chooses tauO each period in autarky)
                # continuation mixes mO' by reentry and mP' by P default decision today.
                iBP_star = int(np.argmin(np.abs(bPgrid - EbP)))
                cont_O_reenter = (1.0 - dP_here) * EVO00[iz2, iBO0, iBP_star] + dP_here * EVO01[iz2, iBO0, iBP0]
                cont_O_stay = (1.0 - dP_here) * EVO10[iz2, iBO0, iBP_star] + dP_here * EVO11[iz2, iBO0, iBP0]
                contO = params.beta * (params.lam_reentry * cont_O_reenter + (1.0 - params.lam_reentry) * cont_O_stay)

                W_tauO = uO_d_tau + v_public(TO_d_tau, params) + contO
                VO_inc, p_tauO = logsumexp_and_probs(W_tauO, params.rho_a)
                VO_here = params.euler_gamma * params.rho_a + VO_inc
                EtO = float(np.sum(p_tauO * tauO))

                # lender mu
                cost = float(np.sum(p_bP * (qP * bPgrid)))
                C_rep = params.y_L + bP_now - cost
                C_def = params.y_L
                C = (1.0 - dP_here) * C_rep + dP_here * C_def
                C = max(C, 1e-12)
                mu_here = C ** (-params.sigma_L)

                VP10[iz2, iBO, iBP] = VP_here
                VO10[iz2, iBO, iBP] = VO_here
                dP10[iz2, iBO, iBP] = dP_here
                mu10[iz2, iBO, iBP] = mu_here

                EbP_rep[iz2, iBO, iBP] = EbP
                EtauP_rep[iz2, iBO, iBP] = EtP
                EtauP_def[iz2, iBO, iBP] = EtP_def
                EtauO_aut[iz2, iBO, iBP] = EtO
                qO_pol[iz2, iBO, iBP] = 0.0
                qP_pol[iz2, iBO, iBP] = float(np.sum(p_bP * qP))

    return dict(VO=VO10, VP=VP10, dO=dO10, dP=dP10, mu=mu10,
                EbP_rep=EbP_rep, EtauP_rep=EtauP_rep, EtauP_def=EtauP_def,
                EtauO_aut=EtauO_aut, qO_pol=qO_pol, qP_pol=qP_pol)


def solve_regime_11(params: Params, grids: Grids, static: StaticBlock, EV_VO, EV_VP):
    """
    (mO,mP) = (1,1): both excluded.
    Each chooses tau in autarky; next m' for each is reentry with lambda.
    """
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    tauO = grids.tauO
    tauP = grids.tauP

    iBO0 = grids.iBO0
    iBP0 = grids.iBP0

    VO11 = np.zeros((NZ2, NB, NB))
    VP11 = np.zeros((NZ2, NB, NB))
    dO11 = np.ones((NZ2, NB, NB))
    dP11 = np.ones((NZ2, NB, NB))
    mu11 = np.full((NZ2, NB, NB), params.y_L ** (-params.sigma_L))

    EtauO_aut = np.zeros((NZ2, NB, NB))
    EtauP_aut = np.zeros((NZ2, NB, NB))
    qO_pol = np.zeros((NZ2, NB, NB))
    qP_pol = np.zeros((NZ2, NB, NB))

    EVO00 = EV_VO[0, 0]; EVO01 = EV_VO[0, 1]; EVO10 = EV_VO[1, 0]; EVO11e = EV_VO[1, 1]
    EVP00 = EV_VP[0, 0]; EVP01 = EV_VP[0, 1]; EVP10 = EV_VP[1, 0]; EVP11e = EV_VP[1, 1]

    for iz2 in range(NZ2):
        uO_d = static.uO_def[iz2, :]; TO_d = static.TO_def[iz2, :]
        uP_d = static.uP_def[iz2, :]; TP_d = static.TP_def[iz2, :]

        # continuation when both excluded: next mO' and mP' independent with lambda
        # debts always 0
        contO = params.beta * (
            params.lam_reentry * params.lam_reentry * EVO00[iz2, iBO0, iBP0] +
            params.lam_reentry * (1.0 - params.lam_reentry) * EVO01[iz2, iBO0, iBP0] +
            (1.0 - params.lam_reentry) * params.lam_reentry * EVO10[iz2, iBO0, iBP0] +
            (1.0 - params.lam_reentry) ** 2 * EVO11e[iz2, iBO0, iBP0]
        )
        contP = params.beta * (
            params.lam_reentry * params.lam_reentry * EVP00[iz2, iBO0, iBP0] +
            params.lam_reentry * (1.0 - params.lam_reentry) * EVP01[iz2, iBO0, iBP0] +
            (1.0 - params.lam_reentry) * params.lam_reentry * EVP10[iz2, iBO0, iBP0] +
            (1.0 - params.lam_reentry) ** 2 * EVP11e[iz2, iBO0, iBP0]
        )

        W_O = uO_d + v_public(TO_d, params) + contO
        incO, pO = logsumexp_and_probs(W_O, params.rho_a)
        VO_here = params.euler_gamma * params.rho_a + incO
        EtO = float(np.sum(pO * tauO))

        W_P = uP_d + v_public(TP_d, params) + contP
        incP, pP = logsumexp_and_probs(W_P, params.rho_a)
        VP_here = params.euler_gamma * params.rho_a + incP
        EtP = float(np.sum(pP * tauP))

        VO11[iz2, :, :] = VO_here
        VP11[iz2, :, :] = VP_here
        EtauO_aut[iz2, :, :] = EtO
        EtauP_aut[iz2, :, :] = EtP

    return dict(VO=VO11, VP=VP11, dO=dO11, dP=dP11, mu=mu11,
                EtauO_aut=EtauO_aut, EtauP_aut=EtauP_aut, qO_pol=qO_pol, qP_pol=qP_pol)


# ============================================================
# 9. Main stationary solver loop
# ============================================================

def solve_stationary_appendixC(params: Params, grids: Grids, Pi2, static: StaticBlock):
    eq = init_eq(params, grids)

    for it in range(params.max_iter):
        # expectations from previous iterate (stationary recursion)
        EV_VO, EV_VP, N_O, N_P = build_expectations(params, grids, Pi2, eq)

        # solve each regime using current expectations
        out00 = solve_regime_00(params, grids, static, EV_VO, EV_VP, N_O, N_P)
        out01 = solve_regime_01(params, grids, static, EV_VO, EV_VP, N_O, N_P)
        out10 = solve_regime_10(params, grids, static, EV_VO, EV_VP, N_O, N_P)
        out11 = solve_regime_11(params, grids, static, EV_VO, EV_VP)

        # assemble new eq arrays
        NZ2 = params.NZ * params.NZ
        NB = grids.bO.size
        shape = (NZ2, NB, NB, 2, 2)

        VO_new = np.zeros(shape)
        VP_new = np.zeros(shape)
        dO_new = np.zeros(shape)
        dP_new = np.zeros(shape)
        mu_new = np.zeros(shape)

        # fill each regime
        VO_new[:, :, :, 0, 0] = out00["VO"]
        VP_new[:, :, :, 0, 0] = out00["VP"]
        dO_new[:, :, :, 0, 0] = out00["dO"]
        dP_new[:, :, :, 0, 0] = out00["dP"]
        mu_new[:, :, :, 0, 0] = out00["mu"]

        VO_new[:, :, :, 0, 1] = out01["VO"]
        VP_new[:, :, :, 0, 1] = out01["VP"]
        dO_new[:, :, :, 0, 1] = out01["dO"]
        dP_new[:, :, :, 0, 1] = out01["dP"]
        mu_new[:, :, :, 0, 1] = out01["mu"]

        VO_new[:, :, :, 1, 0] = out10["VO"]
        VP_new[:, :, :, 1, 0] = out10["VP"]
        dO_new[:, :, :, 1, 0] = out10["dO"]
        dP_new[:, :, :, 1, 0] = out10["dP"]
        mu_new[:, :, :, 1, 0] = out10["mu"]

        VO_new[:, :, :, 1, 1] = out11["VO"]
        VP_new[:, :, :, 1, 1] = out11["VP"]
        dO_new[:, :, :, 1, 1] = out11["dO"]
        dP_new[:, :, :, 1, 1] = out11["dP"]
        mu_new[:, :, :, 1, 1] = out11["mu"]

        # Differences on key objects
        diffV = max(np.max(np.abs(VO_new - eq.VO)), np.max(np.abs(VP_new - eq.VP)))
        diffD = max(np.max(np.abs(dO_new - eq.dO)), np.max(np.abs(dP_new - eq.dP)))
        diffM = np.max(np.abs(mu_new - eq.muL))
        diff = max(diffV, diffD, diffM)

        print(f"iter {it:3d}  diff={diff:.3e}  (V={diffV:.3e}, d={diffD:.3e}, mu={diffM:.3e})")

        # damping update
        eq.VO = (1.0 - params.damp) * eq.VO + params.damp * VO_new
        eq.VP = (1.0 - params.damp) * eq.VP + params.damp * VP_new
        eq.dO = (1.0 - params.damp) * eq.dO + params.damp * dO_new
        eq.dP = (1.0 - params.damp) * eq.dP + params.damp * dP_new
        eq.muL = (1.0 - params.damp) * eq.muL + params.damp * mu_new

        # store policies needed for simulation (all regimes)
        eq.EbO_rep[:, :, :, 0, 0] = out00["EbO_rep"]
        eq.EtauO_rep[:, :, :, 0, 0] = out00["EtauO_rep"]
        eq.EtauO_def[:, :, :, 0, 0] = out00["EtauO_def"]
        eq.EbP_rep[:, :, :, 0, 0] = out00["EbP_rep"]
        eq.EtauP_rep[:, :, :, 0, 0] = out00["EtauP_rep"]
        eq.EtauP_def[:, :, :, 0, 0] = out00["EtauP_def"]

        eq.dP_if_Orep[:, :, :, 0, 0] = out00["dP_if_Orep"]
        eq.EbP_if_Orep[:, :, :, 0, 0] = out00["EbP_if_Orep"]
        eq.EtauP_if_Orep[:, :, :, 0, 0] = out00["EtauP_if_Orep"]
        eq.dP_if_Odef[:, :, :, 0, 0] = out00["dP_if_Odef"]
        eq.EbP_if_Odef[:, :, :, 0, 0] = out00["EbP_if_Odef"]
        eq.EtauP_if_Odef[:, :, :, 0, 0] = out00["EtauP_if_Odef"]

        eq.qO_pol[:, :, :, 0, 0] = out00["qO_pol"]
        eq.qP_pol[:, :, :, 0, 0] = out00["qP_pol"]

        eq.EbO_rep[:, :, :, 0, 1] = out01["EbO_rep"]
        eq.EtauO_rep[:, :, :, 0, 1] = out01["EtauO_rep"]
        eq.EtauO_def[:, :, :, 0, 1] = out01["EtauO_def"]
        eq.EtauP_def[:, :, :, 0, 1] = out01["EtauP_aut"]
        eq.qO_pol[:, :, :, 0, 1] = out01["qO_pol"]
        eq.qP_pol[:, :, :, 0, 1] = out01["qP_pol"]

        eq.EbP_rep[:, :, :, 1, 0] = out10["EbP_rep"]
        eq.EtauP_rep[:, :, :, 1, 0] = out10["EtauP_rep"]
        eq.EtauP_def[:, :, :, 1, 0] = out10["EtauP_def"]
        eq.EtauO_def[:, :, :, 1, 0] = out10["EtauO_aut"]
        eq.qO_pol[:, :, :, 1, 0] = out10["qO_pol"]
        eq.qP_pol[:, :, :, 1, 0] = out10["qP_pol"]

        eq.EtauO_def[:, :, :, 1, 1] = out11["EtauO_aut"]
        eq.EtauP_def[:, :, :, 1, 1] = out11["EtauP_aut"]
        eq.qO_pol[:, :, :, 1, 1] = out11["qO_pol"]
        eq.qP_pol[:, :, :, 1, 1] = out11["qP_pol"]

        if diff < params.tol:
            print("Converged.")
            break

    return eq


# ============================================================
# 10. Simulation with discrete default draws, expected repay policies
# ============================================================

def draw_markov(P, s0, T, rng):
    s = np.empty(T + 1, dtype=int)
    s[0] = s0
    for t in range(T):
        s[t + 1] = rng.choice(P.shape[0], p=P[s[t], :])
    return s
def simulate(params: Params, grids: Grids, Pi2, eq: Eq,
             zO, zP, zO_def, zP_def,
             T=10000, burn=2000, seed=0):
    rng = np.random.default_rng(seed)
    NZ2 = params.NZ * params.NZ
    NB = grids.bO.size
    iBO0, iBP0 = grids.iBO0, grids.iBP0

    # simulate z2 index path
    iz0 = NZ2 // 2
    iz_path = draw_markov(Pi2, iz0, T, rng)

    # init states
    iBO = iBO0
    iBP = iBP0
    mO = 0
    mP = 0

    # storage
    DO = np.zeros(T, dtype=int)
    DP = np.zeros(T, dtype=int)

    bO = np.zeros(T); bP = np.zeros(T)
    bO_p = np.zeros(T); bP_p = np.zeros(T)

    tauO = np.zeros(T); tauP = np.zeros(T)

    # NEW: macro variables
    zO_used = np.zeros(T); zP_used = np.zeros(T)
    lO = np.zeros(T); lP = np.zeros(T)
    yO = np.zeros(T); yP = np.zeros(T)
    cO = np.zeros(T); cP = np.zeros(T)
    TO_tax = np.zeros(T); TP_tax = np.zeros(T)   # tax revenue T = tau*c
    gO = np.zeros(T); gP = np.zeros(T)

    # prices (expected/policy-averaged) + lender consumption
    qO = np.zeros(T); qP = np.zeros(T)
    CL = np.zeros(T)

    for t in range(T):
        iz2 = int(iz_path[t])

        bO[t] = grids.bO[iBO]
        bP[t] = grids.bP[iBP]

        # --- O default draw ---
        if mO == 1:
            DO[t] = 1
        else:
            pO = float(eq.dO[iz2, iBO, iBP, mO, mP])
            DO[t] = 1 if rng.random() < pO else 0

        # O expected policy conditional on repay
        if DO[t] == 1:
            bO_p[t] = 0.0
            tauO[t] = float(eq.EtauO_def[iz2, iBO, iBP, mO, mP])
        else:
            bO_p[t] = float(eq.EbO_rep[iz2, iBO, iBP, mO, mP])
            tauO[t] = float(eq.EtauO_rep[iz2, iBO, iBP, mO, mP])

        # --- P default draw (conditional in (0,0)) ---
        if mP == 1:
            DP[t] = 1
        else:
            if (mO, mP) == (0, 0):
                if DO[t] == 0:
                    pP = float(eq.dP_if_Orep[iz2, iBO, iBP, 0, 0])
                else:
                    pP = float(eq.dP_if_Odef[iz2, iBO, iBP, 0, 0])
            else:
                pP = float(eq.dP[iz2, iBO, iBP, mO, mP])
            DP[t] = 1 if rng.random() < pP else 0

        # P expected policy conditional on repay
        if DP[t] == 1:
            bP_p[t] = 0.0
            tauP[t] = float(eq.EtauP_def[iz2, iBO, iBP, mO, mP])
        else:
            if (mO, mP) == (0, 0):
                if DO[t] == 0:
                    bP_p[t] = float(eq.EbP_if_Orep[iz2, iBO, iBP, 0, 0])
                    tauP[t] = float(eq.EtauP_if_Orep[iz2, iBO, iBP, 0, 0])
                else:
                    bP_p[t] = float(eq.EbP_if_Odef[iz2, iBO, iBP, 0, 0])
                    tauP[t] = float(eq.EtauP_if_Odef[iz2, iBO, iBP, 0, 0])
            else:
                bP_p[t] = float(eq.EbP_rep[iz2, iBO, iBP, mO, mP])
                tauP[t] = float(eq.EtauP_rep[iz2, iBO, iBP, mO, mP])

        # prices used in simulation (regime-specific policy-averaged prices)
        qO[t] = float(eq.qO_pol[iz2, iBO, iBP, mO, mP])
        qP[t] = float(eq.qP_pol[iz2, iBO, iBP, mO, mP])

        # ------------------------------------------------------------
        # NEW: allocations using GHH block
        # ------------------------------------------------------------
        # choose productivity depending on default
        zO_used[t] = zO_def[iz2] if DO[t] == 1 else zO[iz2]
        zP_used[t] = zP_def[iz2] if DP[t] == 1 else zP[iz2]

        # labor from intratemporal FOC: l = ( z / (theta*(1+tau)) )^(1/nu)
        lO[t] = (zO_used[t] / (params.theta * (1.0 + tauO[t]))) ** (1.0 / params.nu)
        lP[t] = (zP_used[t] / (params.theta * (1.0 + tauP[t]))) ** (1.0 / params.nu)

        # output
        yO[t] = zO_used[t] * lO[t]
        yP[t] = zP_used[t] * lP[t]

        # private consumption: c = y/(1+tau)
        cO[t] = yO[t] / (1.0 + tauO[t])
        cP[t] = yP[t] / (1.0 + tauP[t])

        # tax revenue: T = tau * c
        TO_tax[t] = tauO[t] * cO[t]
        TP_tax[t] = tauP[t] * cP[t]

        # public goods via gov budget:
        if DO[t] == 0:
            gO[t] = TO_tax[t] + qO[t] * bO_p[t] - bO[t]
        else:
            gO[t] = TO_tax[t]

        if DP[t] == 0:
            gP[t] = TP_tax[t] + qP[t] * bP_p[t] - bP[t]
        else:
            gP[t] = TP_tax[t]

        # lender consumption accounting (expected-policy version)
        CL[t] = params.y_L + (1 - DO[t]) * bO[t] + (1 - DP[t]) * bP[t] - qO[t] * bO_p[t] - qP[t] * bP_p[t]

        # ------------------------------------------------------------
        # update market access (discrete)
        # ------------------------------------------------------------
        if mO == 0 and DO[t] == 1:
            mO_next = 1
        elif mO == 1:
            mO_next = 0 if rng.random() < params.lam_reentry else 1
        else:
            mO_next = 0

        if mP == 0 and DP[t] == 1:
            mP_next = 1
        elif mP == 1:
            mP_next = 0 if rng.random() < params.lam_reentry else 1
        else:
            mP_next = 0

        # snap expected b' to grid for next state index
        iBO = int(np.argmin(np.abs(grids.bO - bO_p[t])))
        iBP = int(np.argmin(np.abs(grids.bP - bP_p[t])))

        mO, mP = mO_next, mP_next

    sl = slice(burn, T)
    return {
        "DO": DO[sl], "DP": DP[sl],
        "bO": bO[sl], "bP": bP[sl],
        "bO_p": bO_p[sl], "bP_p": bP_p[sl],
        "tauO": tauO[sl], "tauP": tauP[sl],
        "zO": zO_used[sl], "zP": zP_used[sl],
        "lO": lO[sl], "lP": lP[sl],
        "yO": yO[sl], "yP": yP[sl],
        "cO": cO[sl], "cP": cP[sl],
        "TO_tax": TO_tax[sl], "TP_tax": TP_tax[sl],
        "gO": gO[sl], "gP": gP[sl],
        "qO": qO[sl], "qP": qP[sl],
        "CL": CL[sl],
    }



# ============================================================
# 11. Event study around O default
# ============================================================

def event_study_around_defaults(sim, window=10):
    """
    Build event-time averages around O default events.
    window=20 means [-10,...,0,...,+10]
    """
    DO = sim["DO"]
    T = DO.size
    events = np.where(DO == 1)[0]
    # exclude events too close to edges
    events = events[(events >= window) & (events < T - window)]
    if events.size == 0:
        return None

    def stack_series(x):
        X = np.stack([x[e - window:e + window + 1] for e in events], axis=0)
        return X.mean(axis=0)

    out = {"event_time": np.arange(-window, window + 1)}
    for k, v in sim.items():
        if v.ndim == 1 and v.size == T:
            out[k] = stack_series(v.astype(float))
    out["num_events"] = events.size
    return out


# ============================================================
# 12. Run
# ============================================================

if __name__ == "__main__":
    params = Params()
    grids = make_grids_10()

    zO, zP, zO_def, zP_def, Pi2 = build_joint_z(params)
    static = precompute_static(params, grids, zO, zP, zO_def, zP_def)

    eq = solve_stationary_appendixC(params, grids, Pi2, static)

    sim = simulate(params, grids, Pi2, eq, zO, zP, zO_def, zP_def,
                   T=10000, burn=2000, seed=123)

    ev = event_study_around_defaults(sim, window=10)
    if ev is not None:
        print("Event study computed with", ev["num_events"], "events.")
        idx0 = 10  # event time 0 location for window=10
        print("Avg P default at t=0:", ev["DP"][idx0])
        print("Avg qP at t=0:", ev["qP"][idx0])
        print("Avg yP at t=0:", ev["yP"][idx0])

   
