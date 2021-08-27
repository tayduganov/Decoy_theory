from numpy import exp, log, log2, sqrt, arange
# import numpy as np
from scipy.stats import norm
from math import ceil, fabs
from collections import OrderedDict
import random

# -----------------------------------------------------------------------------
# NOTATIONS
# -----------------------------------------------------------------------------
# mu,nu1,nu2 = signal, weak and vacuum decoy pulse intensities out of Alice
# p_{mu,nu1,nu2} = probabilities to generate pulses of given intensities
# loss_qc = total losses in quantum channel (fiber, connectors, etc.) [dB]
# loss_B = internal losses of Bob [dB]
# eta = detector's efficiency
# tau_d = detector's dead time
# p_dc = DCR*t_gate = dark count probability (per detector per gate)
# p_ap = cumulative afterpulse probability since the end of dead time
# p_opt = (1-V)/2 = probability of a photon to hit a wrong detector
# f = pulse repetition frequency [Hz]
# N_{mu,nu1,nu2} = number of sent pulses (of same bases)
# M_{mu,nu1,nu2} = number of measured pulses (of same bases)
# Q_{mu,nu1,nu2} = M_{mu,nu1,nu2} / N_{mu,nu1,nu2} = gain
# E_{mu,nu1,nu2} = M_{mu,nu1,nu2}_error / M_{mu,nu1,nu2} = QBER
# l_ver = verified key length of LDPC block
# - if all frames in the block are corrected, l_ver = l_block
# - if some frames cannot be corrected, they are discarded, and hence
#   l_ver = l_frame * number of corrected frames
# (our FER~10^-4 => for theoretical estimations one can assume l_ver=l_block)
# f_ec = error correction efficiency of LDPC codes, depends on QBER

# -----------------------------------------------------------------------------
# SOME IMPORTANT NOTES
# -----------------------------------------------------------------------------
# 1) Privacy amplification is done on the measured key of length M_mu
# => modify Y1_l, Q1_l, theta1_l and E1_u appropriately:
# - if M_mu is statistical, use upper bound on Q_mu(math expectation of Q_mu)
#   for TKF(ZZRM) method as in the paper;
# - if M_mu is fixed to be l_ver(~l_block), use Q_mu_u=M_mu/N_mu and
#   MathExp_u[Q_mu]=M_mu/N_mu in the functions

# 2) If Q1_l/Q_mu or E1_u estimations turn out to be non-physical:
# - Q1_l/Q_mu<0, set 0
# - Q1_l/Q_mu>=1, set 0 (impossible to have ALL single-photon pulses)
# - E1_u<=0, set 0.5
# - E1_u>0.5, set 0.5 (in this case one should estimate E1_l instead of E1_u)
# Any condition above will make Q1_l*[1-h2(E1_u)]=0 and hence
# R_sec= -R_ver*f_ec*h2(E_mu)<0 => this block will be discarded.

# 3) For technical reasons we cannot set nu<mu/100.
# -----------------------------------------------------------------------------


# Shannon information entropy
def h2(x):
    if (x == 0.0 or x == 1.0):
        return 0.0
    else:
        return -x * log2(x) - (1 - x) * log2(1 - x)


# -----------------------------------------------------------------------------
# LDPC-based error correction
# -----------------------------------------------------------------------------

# code rate adaptation law based on the E_mu estimation a priori
def choose_code_rate(E_mu, code_rate_change, frame_len, alpha, f_start):
    r, s, p = 0., 0, 0
    max_puncturable_table = {0.5: 3440, 0.55: 3204, 0.6: 2771, 0.65: 2537,
                             0.7: 1848, 0.75: 1134, 0.8: 1494, 0.85: 1068,
                             0.9: 764}
    desired_code_rate = 1. - h2(E_mu) * f_start
    s_p_count = int(alpha * frame_len)
    suitable_codes = dict()
    for code_rate in code_rate_change:
        code_p = int(ceil(frame_len * (1. - code_rate
                     - f_start * (1. - alpha) * h2(E_mu))))
        code_s = int(s_p_count - code_p)
        max_puncturable_count = max_puncturable_table[code_rate]
        if code_s >= 0 and 0 <= code_p <= max_puncturable_count:
            suitable_codes[code_rate] = (code_s, code_p)
    if len(suitable_codes):
        od = OrderedDict(sorted(suitable_codes.items()))
        rate_params = suitable_codes[next(reversed(od))]
        r, s, p = next(reversed(od)), rate_params[0], rate_params[1]
    else:
        code_rate_min = min(code_rate_change)
        if desired_code_rate < code_rate_min:
            r, s, p = code_rate_min, s_p_count, 0
        else:
            r = max(code_rate_change)
            p = min(max_puncturable_table[r], s_p_count)
            s = int(s_p_count - p)
    return r, int(s), int(p)


def add_round_disclosed_count(f_cur, f_target, frame_len, E_mu):
    delta_f = 0.1
    if f_cur != f_target:
        delta_f = fabs(f_cur - f_target)
    return int(ceil(delta_f * frame_len * E_mu))


def f_evaluate(m, p, payload_len, E_mu, add_disclosed):
    return float(m - p + add_disclosed) / float(payload_len * h2(E_mu))


# theoretical f_ec estimation
def f_ec_th(E_mu):
    # current LDPC codes pool parameters
    code_rate_change = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    # additional round probability
    add_round_probs = {0.5: 0.05, 0.55: 0.05, 0.6: 0.1, 0.65: 0.15, 0.7: 0.2,
                       0.75: 0.5, 0.8: 0.2, 0.85: 0.25, 0.9: 0.5}
    f_start = 1.15
    f_stop = 3.0
    f_step = 0.03
    f_pat = arange(f_start, f_stop, f_step)  # target efficiency pattern
    frame_len = 8000
    alpha = 0.15
    r, s, p = choose_code_rate(E_mu, code_rate_change, frame_len, alpha,
                               f_start)
    m = int(ceil((1. - r) * frame_len))  # syndrome length
    add_disclosed = 0
    payload_len = frame_len - p - s
    f_cur = 0.
    add_round_prob = add_round_probs[r]
    for f_target in f_pat:
        f_cur = f_evaluate(m, p, payload_len, E_mu, add_disclosed)
        if random.uniform(0, 1.) > add_round_prob:
            break
        add_disclosed += add_round_disclosed_count(f_cur, f_target, frame_len,
                                                   E_mu)
    return f_cur


# -----------------------------------------------------------------------------
# ideal purely-theoretical predictions/expectations
# Y, Q, E can be found in e.g. Ma, Qi, Zhao, Lo, arXiv:0503005
# -----------------------------------------------------------------------------

# single-photon detection (=click) probability
def p1_sig(eta, loss_qc, loss_B):
    return eta * 10 ** (-(loss_qc + loss_B) / 10)


# signal pulse detection (=click) probability
def p_sig(mu, eta, loss_qc, loss_B):
    return 1 - exp(-mu * eta * 10 ** (-(loss_qc + loss_B) / 10))


# overall gain per pulse
def Q_th(mu, p_dc, p_ap, *pars):
    return (2 * p_dc + p_sig(mu, *pars)) * (1 + p_ap)


# overall QBER per pulse
def E_th(mu, p_opt, p_dc, p_ap, *pars):
    return (p_dc + p_opt * p_sig(mu, *pars)) / Q_th(mu, p_dc, p_ap, *pars)


# single-photon gain per pulse
def Q1_th(mu, p_dc, p_ap, eta, loss_qc, loss_B):
    return (2 * p_dc + p1_sig(eta, loss_qc, loss_B)) * mu * exp(-mu) * \
        (1 + p_ap)


# single-photon QBER per pulse
def E1_th(mu, p_opt, p_dc, p_ap, eta, loss_qc, loss_B):
    return (p_dc + p_opt * p1_sig(eta, loss_qc, loss_B)) / \
            (2 * p_dc + p1_sig(eta, loss_qc, loss_B))


# sifted key generation rate (taking into account detector's dead time)
def R_sift_nodecoy_th(mu, f, tau_d, *pars):
    return f * Q_th(mu, *pars) / 2 / (1 + f * Q_th(mu, *pars) * tau_d)


def R_sift_th(mu, nu1, nu2, p_mu, p_nu1, f, tau_d, *pars):
    R_mu = f * p_mu * Q_th(mu, *pars)
    R_tot = f * (p_mu * Q_th(mu, *pars) + p_nu1 * Q_th(nu1, *pars)
                 + (1 - p_mu - p_nu1) * Q_th(nu2, *pars))
    return R_mu / 2 / (1 + R_tot * tau_d)


# privacy amplification coefficient ("ideal" GLLP)
def r_pa_th(mu, f_ec, p_opt, *pars):
    return Q1_th(mu, *pars) / Q_th(mu, *pars) * \
        (1 - h2(E1_th(mu, p_opt, *pars))) - f_ec * h2(E_th(mu, p_opt, *pars))


# limit case for ideal single-photon source and perfect error correction codes
def r_pa_limit(*pars):
    return 1 - 2 * h2(E_th(*pars))


# -----------------------------------------------------------------------------
# extract p_dc/(t*eta) and p_opt from experimental QBERs
# -----------------------------------------------------------------------------


def p_dc_over_etat_exp(nu1, nu2, E_nu1, E_nu2,):
    return (E_nu2 - E_nu1) * nu1 * nu2 / \
        (nu1 * (1 - 2 * E_nu2) - nu2 * (1 - 2 * E_nu1))


def p_opt_exp(nu1, nu2, E_nu1, E_nu2):
    return (E_nu1 * nu1 * (1 - 2 * E_nu2) - E_nu2 * nu2 * (1 - 2 * E_nu1)) / \
        (nu1 * (1 - 2 * E_nu2) - nu2 * (1 - 2 * E_nu1))


'''def p_dc_exp(nu1, nu2, E_nu1, E_nu2, eta, loss_qc, loss_B):
    xx = eta * 10 ** (-(loss_qc + loss_B) / 10)
    return (1 - exp(-nu1 * xx)) * (1 - exp(-nu2 * xx)) * (E_nu2 - E_nu1) / \
        ((1 - 2 * E_nu1) * exp(-nu2 * xx)
         - (1 - 2 * E_nu2) * exp(-nu1 * xx)
         - 2 * (E_nu2 - E_nu1))


def p_opt_exp(nu1, nu2, E_nu1, E_nu2, eta, loss_qc, loss_B):
    xx = eta * 10 ** (-(loss_qc + loss_B) / 10)
    return ((1 - 2 * E_nu1) * E_nu2 * exp(-nu2 * xx)
            - (1 - 2 * E_nu2) * E_nu1 * exp(-nu1 * xx)
            - (E_nu2 - E_nu1)) / \
        ((1 - 2 * E_nu1) * exp(-nu2 * xx)
         - (1 - 2 * E_nu2) * exp(-nu1 * xx)
         - 2 * (E_nu2 - E_nu1))'''


# -----------------------------------------------------------------------------
# NO DECOY
# Eve blocks such a fraction of single-photon pulses that Bob gets the expected
# statistics (Q_mu=Q_mu^th) and leaves at least 1 photon from multi-photon
# pulses for herself and sends the others to Bob via lossless channel.
# Since Alice and Bob cannot distinguish Y0 and Y1, they have to consider these
# two states together (Ma, PhD thesis p.32, arXiv:0808.1385)
# -----------------------------------------------------------------------------

# PNS "Eve allmighty" strategy:
# -Eve can break into Bob's device and control detectors making eta=100%
# -doesn't depend on the number of left/sent photons

# lower bound on Q1+Y0
def Q1_l_nodecoy_PNS_allmighty(mu, Q_mu):
    return Q_mu - 1 + (1 + mu) * exp(-mu)


# upper bound on single-photon QBER
def E1_u_nodecoy_PNS_allmighty(mu, E_mu, Q_mu):
    return E_mu * Q_mu / Q1_l_nodecoy_PNS_allmighty(mu, Q_mu)


# lower bound on the secret key generation rate
def R_sec_nodecoy_PNS_allmighty(mu, f_ec, R_sift, E_mu, Q_mu):
    return (Q1_l_nodecoy_PNS_allmighty(mu, Q_mu) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_allmighty(mu, E_mu, Q_mu)))
            - f_ec * h2(E_mu)) * R_sift


# lower bound on the secret key length
def l_sec_nodecoy_PNS_allmighty(mu, f_ec, l_ver, E_mu, Q_mu):
    return (Q1_l_nodecoy_PNS_allmighty(mu, Q_mu) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_allmighty(mu, E_mu, Q_mu)))
            - f_ec * h2(E_mu)) * l_ver


# PNS optimal strategy:
# -Eve cannot break into Bob's module and control detectors
# -Eve leaves 1 photon and sends n-1 to Bob

def Q1_l_nodecoy_PNS_opt(mu, Q_mu, eta, loss_B):
    t_B = eta * 10 ** (-loss_B / 10)
    if t_B == 1:
        return Q_mu - 1 + (1 + mu) * exp(-mu)  # same as in allmighty scenario
    else:
        return Q_mu - 1 + (exp(-t_B * mu) - t_B * exp(-mu)) / (1 - t_B)


def E1_u_nodecoy_PNS_opt(mu, E_mu, Q_mu, *pars):
    return E_mu * Q_mu / Q1_l_nodecoy_PNS_opt(mu, Q_mu, *pars)


def R_sec_nodecoy_PNS_opt(mu, f_ec, R_sift, E_mu, Q_mu, *pars):
    return (Q1_l_nodecoy_PNS_opt(mu, Q_mu, *pars) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_opt(mu, E_mu, Q_mu, *pars)))
            - f_ec * h2(E_mu)) * R_sift


def l_sec_nodecoy_PNS_opt(mu, f_ec, l_ver, E_mu, Q_mu, *pars):
    return (Q1_l_nodecoy_PNS_opt(mu, Q_mu, *pars) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_opt(mu, E_mu, Q_mu, *pars)))
            - f_ec * h2(E_mu)) * l_ver


# PNS suboptimal strategy:
# -Eve cannot break into Bob's module and control detector
# -Eve leaves n-1 photons and sends 1 to Bob

def Q1_l_nodecoy_PNS_subopt(mu, Q_mu, eta, loss_B):
    return Q_mu - (1 - (1 + mu) * exp(-mu)) * eta * 10 ** (-loss_B / 10)


def E1_u_nodecoy_PNS_subopt(mu, E_mu, Q_mu, *pars):
    return E_mu * Q_mu / Q1_l_nodecoy_PNS_subopt(mu, Q_mu, *pars)


def R_sec_nodecoy_PNS_subopt(mu, f_ec, R_sift, E_mu, Q_mu, *pars):
    return (Q1_l_nodecoy_PNS_subopt(mu, Q_mu, *pars) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_subopt(mu, E_mu, Q_mu, *pars)))
            - f_ec * h2(E_mu)) * R_sift


def l_sec_nodecoy_PNS_subopt(mu, f_ec, l_ver, E_mu, Q_mu, *pars):
    return (Q1_l_nodecoy_PNS_subopt(mu, Q_mu, *pars) / Q_mu
            * (1 - h2(E1_u_nodecoy_PNS_subopt(mu, E_mu, Q_mu, *pars)))
            - f_ec * h2(E_mu)) * l_ver


# -----------------------------------------------------------------------------
# DECOY without statistic fluctuations
# Ma, Qi, Zhao, Lo, arXiv:0503005
# -----------------------------------------------------------------------------

# lower bound on background yield
def Y0_l_MQZL(nu1, nu2, Q_nu1, Q_nu2):
    y0_tmp = (nu1 * Q_nu2 * exp(nu2) - nu2 * Q_nu1 * exp(nu1)) / (nu1 - nu2)
    if y0_tmp > 0:
        return y0_tmp
    else:
        return 0


# lower bound on single-photon yield
def Y1_l_MQZL(mu, nu1, nu2, Q_mu, Q_nu1, Q_nu2):
    return mu / (nu1 - nu2) / (mu - nu1 - nu2) * \
            (Q_nu1 * exp(nu1) - Q_nu2 * exp(nu2)
             - (nu1 ** 2 - nu2 ** 2) / mu ** 2
             * (Q_mu * exp(mu) - Y0_l_MQZL(nu1, nu2, Q_nu1, Q_nu2)))


# fraction of single-photon pulses
def kappa1_l_MQZL(mu, nu1, nu2, Q_mu, Q_nu1, Q_nu2):
    return mu * exp(-mu) * Y1_l_MQZL(mu, nu1, nu2, Q_mu, Q_nu1, Q_nu2) / Q_mu


# upper bound on single-photon QBER
def E1_u_MQZL(mu, nu1, nu2, E_nu1, E_nu2, Q_mu, Q_nu1, Q_nu2):
    return (E_nu1 * Q_nu1 * exp(nu1) - E_nu2 * Q_nu2 * exp(nu2)) / \
            (nu1 - nu2) / Y1_l_MQZL(mu, nu1, nu2, Q_mu, Q_nu1, Q_nu2)


# lower bound on the secret key generation rate
def R_sec_MQZL(mu, nu1, nu2, f_ec, R_sift, E_mu, E_nu1, E_nu2, *pars):
    return (kappa1_l_MQZL(mu, nu1, nu2, *pars)
            * (1 - h2(E1_u_MQZL(mu, nu1, nu2, E_nu1, E_nu2, *pars)))
            - f_ec * h2(E_mu)) * R_sift


# secret key length, to be obtained from M_mu=Q_mu*N_mu
def l_sec_MQZL(mu, nu1, nu2, f_ec, l_ver, E_mu, E_nu1, E_nu2, *pars):
    return (kappa1_l_MQZL(mu, nu1, nu2, *pars)
            * (1 - h2(E1_u_MQZL(mu, nu1, nu2, E_nu1, E_nu2, *pars)))
            - f_ec * h2(E_mu)) * l_ver


def r_pa_MQZL(mu, nu1, nu2, f_ec, E_mu, E_nu1, E_nu2, *pars):
    return kappa1_l_MQZL(mu, nu1, nu2, *pars) * \
        (1 - h2(E1_u_MQZL(mu, nu1, nu2, E_nu1, E_nu2, *pars))) - \
        f_ec * h2(E_mu)


# -----------------------------------------------------------------------------
# DECOY with statistic fluctuations (based on CLT)
# Trushechkin, Kiktenko, Fedorov, arXiv:1702.08531
# -----------------------------------------------------------------------------

# ppf=quantile: prob(rand.norm.variable > mean + ppf(p)*sigma)=1-p
def phi(eps):
    return norm.ppf(1 - eps / 7)


# lower and upper bound on gains (assuming Q_exp~=<Q>=Q_th)
def Q_l_TKF(eps, M_x, N_x):
    Q_x = M_x / N_x  # M is random, N is known and treated as non-random
    return Q_x - phi(eps) * sqrt(Q_x * (1 - Q_x) / N_x)


def Q_u_TKF(eps, M_x, N_x):
    Q_x = M_x / N_x
    return Q_x + phi(eps) * sqrt(Q_x * (1 - Q_x) / N_x)


def E_u_TKF(eps, E_x, M_x):
    return E_x + phi(eps) * sqrt(E_x * (1 - E_x) / M_x)


def Y0_l_TKF(nu1, nu2, eps, M_nu1, M_nu2, N_nu1, N_nu2):
    y0_tmp = (nu1 * Q_l_TKF(eps, M_nu2, N_nu2) * exp(nu2)
              - nu2 * Q_u_TKF(eps, M_nu1, N_nu1) * exp(nu1)) / (nu1 - nu2)
    if y0_tmp > 0:
        return y0_tmp
    else:
        return 0


def Q1_l_TKF(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1, N_nu2):
    return mu ** 2 * exp(-mu) / (nu1 - nu2) / (mu - nu1 - nu2) * \
            (Q_l_TKF(eps, M_nu1, N_nu1) * exp(nu1)
             - Q_u_TKF(eps, M_nu2, N_nu2) * exp(nu2)
             - (nu1 ** 2 - nu2 ** 2) / mu ** 2
             * (Q_u_TKF(eps, M_mu, N_mu) * exp(mu)
             # * (M_mu / N_mu * exp(mu)
                - Y0_l_TKF(nu1, nu2, eps, M_nu1, M_nu2, N_nu1, N_nu2)))


def theta1_l_TKF(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, *pars):
    theta1_tmp = Q1_l_TKF(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu,
                          *pars) / Q_u_TKF(eps, M_mu, N_mu)  # (M_mu / N_mu)
    if theta1_tmp <= 0:
        print('WARNING: theta1_l<=0')
        return 0
    elif theta1_tmp >= 1:
        print('WARNING: theta1_l>=1')
        return 0
    else:
        return theta1_tmp


def kappa1_l_TKF(mu, nu1, nu2, eps, l_ver, *pars):
    theta1_tmp = theta1_l_TKF(mu, nu1, nu2, eps, *pars)
    return theta1_tmp - phi(eps) * sqrt(theta1_tmp * (1 - theta1_tmp) / l_ver)


def E1_u_TKF(mu, nu1, nu2, eps, l_ver, E_mu, M_mu, M_nu1, M_nu2, N_mu, *pars):
    e0Y0 = exp(-mu) * Y0_l_TKF(nu1, nu2, eps, M_nu1, M_nu2, *pars) / 2
    v_tmp = N_mu * e0Y0 - phi(eps) * sqrt(N_mu * e0Y0 * (1 - e0Y0))
    e1_tmp = (E_mu - v_tmp / l_ver) / \
        kappa1_l_TKF(mu, nu1, nu2, eps, l_ver, M_mu, M_nu1, M_nu2, N_mu, *pars)
    if e1_tmp <= 0:
        print('WARNING: E1_u<=0')
        return 0.5
    elif e1_tmp >= 0.5:
        print('WARNING: E1_u>=0.5')
        return 0.5
    else:
        return e1_tmp


'''
# assuming that errors are also binomially distributed as in the old code
def E1_u_TKF(mu, nu1, nu2, eps, E_mu, M_mu, M_nu1, M_nu2, N_mu, *pars):
    e1_tmp = (E_u_TKF(eps, E_mu, M_mu) * M_mu / N_mu
              # * Q_u_TKF(eps, M_mu, N_mu)
              - Y0_l_TKF(nu1, nu2, eps, M_nu1, M_nu2, *pars) / 2
              * exp(-mu)) / \
          Q1_l_TKF(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, *pars)
    if e1_tmp <= 0:
        print('WARNING: E1_u<=0')
        return 0.5
    elif e1_tmp >= 0.5:
        print('WARNING: E1_u>=0.5')
        return 0.5
    else:
        return e1_tmp + phi(eps) * sqrt(e1_tmp * (1 - e1_tmp) / M_mu)
'''


def l_sec_TKF(mu, nu1, nu2, eps, f_ec, l_ver, E_mu, *pars):
    if (mu < 1 and nu1 < 1 and nu2 < 1
            and mu > nu1 + nu2
            and nu1 > 2 * nu2
            and nu2 >= mu / 100):
        return (kappa1_l_TKF(mu, nu1, nu2, eps, l_ver, *pars)
                * (1 - h2(E1_u_TKF(mu, nu1, nu2, eps, l_ver, E_mu, *pars)))
                - f_ec * h2(E_mu)) * l_ver + 5 * log2(eps)
    else:
        return -100500


def r_pa_TKF(mu, nu1, nu2, eps, f_ec, l_ver, E_mu, *pars):
    if (mu < 1 and nu1 < 1 and nu2 < 1
            and mu > nu1 + nu2
            and nu1 > 2 * nu2
            and nu2 >= mu / 100):
        return kappa1_l_TKF(mu, nu1, nu2, eps, l_ver, *pars) * \
            (1 - h2(E1_u_TKF(mu, nu1, nu2, eps, l_ver, E_mu, *pars))) - \
            f_ec * h2(E_mu)
    else:
        return -100500


def R_sec_TKF(mu, nu1, nu2, eps, f_ec, l_ver, R_ver, *pars):
    return R_ver * r_pa_TKF(mu, nu1, nu2, eps, f_ec, l_ver, *pars)


# -----------------------------------------------------------------------------
# DECOY with statistic fluctuations (based on Chernoff bound)
# Zhang, Zhao, Razavi, Ma, arXiv:1611.02524
# -----------------------------------------------------------------------------

# here X={M_x,E_x*M_x}; delta_lu<1 => valid for X>-6ln(eps/2)!!!
def delta_lu(eps, X):
    return (-3 * log(eps / 2) + sqrt(log(eps / 2) ** 2
            - 8 * log(eps / 2) * X)) / 2 / (X + log(eps / 2))


# here X=math.expectation[M1]=p_1*M1_l
def delta_M1(eps, X):
    return (-log(eps / 2) + sqrt(log(eps / 2) ** 2
            - 8 * log(eps / 2) * X)) / 2 / X


# math expectation lower bound
def MathExp_l(eps, X):
    return X / (1 + delta_lu(eps, X))


# math expectation upper bound
def MathExp_u(eps, X):
    return X / (1 - delta_lu(eps, X))


def Y0_l_ZZRM(nu1, nu2, eps, M_nu1, M_nu2, N_nu1, N_nu2):
    y0_tmp = (nu1 * MathExp_l(eps, M_nu2) / N_nu2 * exp(nu2)
              - nu2 * MathExp_u(eps, M_nu1) / N_nu1 * exp(nu1)) / (nu1 - nu2)
    if y0_tmp > 0:
        return y0_tmp
    else:
        return 0


def Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1, N_nu2):
    return mu / (nu1 - nu2) / (mu - nu1 - nu2) * \
            (MathExp_l(eps, M_nu1) / N_nu1 * exp(nu1)
             - MathExp_u(eps, M_nu2) / N_nu2 * exp(nu2)
             - (nu1 ** 2 - nu2 ** 2) / mu ** 2
             * (MathExp_u(eps, M_mu) / N_mu * exp(mu)
             # * (M_mu / N_mu * exp(mu)
                - Y0_l_ZZRM(nu1, nu2, eps, M_nu1, M_nu2, N_nu1, N_nu2)))


# lower bound on number of detected signal single-photon pulses
def M1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, *pars):
    m1_tmp = Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, *pars) * \
        N_mu * mu * exp(-mu)
    return m1_tmp * (1 - delta_M1(eps, m1_tmp))


def E1_u_ZZRM(mu, nu1, nu2, eps, E_nu1, E_nu2, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
              N_nu2):
    e1_tmp = (MathExp_u(eps, E_nu1 * M_nu1) / N_nu1 * exp(nu1)
              - MathExp_l(eps, E_nu2 * M_nu2) / N_nu2 * exp(nu2)) / \
             (nu1 - nu2) / Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2,
                                     N_mu, N_nu1, N_nu2)
    if e1_tmp <= 0:
        print('WARNING: E1_u<=0', mu, nu1, nu2, M_mu / N_mu, M_nu1 / N_nu1)
        return 0.5
    elif e1_tmp >= 0.5:
        print('WARNING: E1_u>=0.5', mu, nu1, nu2, M_mu / N_mu, M_nu1 / N_nu1)
        return 0.5
    else:
        return e1_tmp


def kappa1_l_ZZRM(mu, nu1, nu2, eps, M_mu, *pars):
    kappa1_tmp = M1_l_ZZRM(mu, nu1, nu2, eps, M_mu, *pars) / M_mu
    if kappa1_tmp <= 0:
        print('WARNING: M1_l/M_mu<=0', mu, nu1, nu2)
        return 0
    elif kappa1_tmp >= 1:
        print('WARNING: M1_l/M_mu>=1', mu, nu1, nu2)
        return 0
    else:
        return kappa1_tmp


def l_sec_ZZRM(mu, nu1, nu2, eps, f_ec, E_mu, E_nu1, E_nu2, M_mu, M_nu1,
               M_nu2, N_mu, N_nu1, N_nu2):
    if (mu < 1 and nu1 < 1 and nu2 < 1
            and mu > nu1 + nu2
            and nu1 > 2 * nu2
            and nu2 >= mu / 100
            and M_mu > -6 * log(eps / 2)
            and M_nu1 > -6 * log(eps / 2)
            and M_nu2 > -6 * log(eps / 2)
            and E_nu1 * M_nu1 > -6 * log(eps / 2)
            and E_nu2 * M_nu2 >= -6 * log(eps / 2)
            and Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
                          N_nu2) * N_mu * mu * exp(-mu) >= -3 * log(eps / 2)):
        return M1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
                         N_nu2) * \
            (1 - h2(E1_u_ZZRM(mu, nu1, nu2, eps, E_nu1, E_nu2, M_mu, M_nu1,
             M_nu2, N_mu, N_nu1, N_nu2))) - \
            M_mu * f_ec * h2(E_mu) + 5 * log2(eps)
    else:
        if M_mu <= -6 * log(eps / 2):
            print('WARNING: M_mu<=-6ln(eps/2)', mu, nu1, nu2)
        if M_nu1 <= -6 * log(eps / 2):
            print('WARNING: M_nu1<=-6ln(eps/2)', mu, nu1, nu2)
        if M_nu2 <= -6 * log(eps / 2):
            print('WARNING: M_nu2<=-6ln(eps/2)', mu, nu1, nu2)
        if E_nu1 * M_nu1 <= -6 * log(eps / 2):
            print('WARNING: E_nu1*M_nu1<=-6ln(eps/2)', mu, nu1, nu2)
        if E_nu2 * M_nu2 < -6 * log(eps / 2):
            print('WARNING: E_nu2*M_nu2<-6ln(eps/2)', mu, nu1, nu2)
        if Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
                     N_nu2) * N_mu * mu * exp(-mu) < -3 * log(eps / 2):
            print('WARNING: p1*M1_l<-3ln(eps/2)', mu, nu1, nu2)
        return -100500


def r_pa_ZZRM(mu, nu1, nu2, eps, f_ec, E_mu, E_nu1, E_nu2, M_mu, M_nu1,
              M_nu2, N_mu, N_nu1, N_nu2):
    if (mu < 1 and nu1 < 1 and nu2 < 1
            and mu > nu1 + nu2
            and nu1 > 2 * nu2
            and nu2 >= mu / 100
            and M_mu > -6 * log(eps / 2)
            and M_nu1 > -6 * log(eps / 2)
            and M_nu2 > -6 * log(eps / 2)
            and E_nu1 * M_nu1 > -6 * log(eps / 2)
            and E_nu2 * M_nu2 >= -6 * log(eps / 2)
            and Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
                          N_nu2) * N_mu * mu * exp(-mu) >= -3 * log(eps / 2)):
        return kappa1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu,
                             N_nu1, N_nu2) * \
            (1 - h2(E1_u_ZZRM(mu, nu1, nu2, eps, E_nu1, E_nu2, M_mu, M_nu1,
                    M_nu2, N_mu, N_nu1, N_nu2))) - f_ec * h2(E_mu)
    else:
        if M_mu <= -6 * log(eps / 2):
            print('WARNING: M_mu<=-6ln(eps/2)', mu, nu1, nu2)
        if M_nu1 <= -6 * log(eps / 2):
            print('WARNING: M_nu1<=-6ln(eps/2)', mu, nu1, nu2)
        if M_nu2 <= -6 * log(eps / 2):
            print('WARNING: M_nu2<=-6ln(eps/2)', mu, nu1, nu2)
        if E_nu1 * M_nu1 <= -6 * log(eps / 2):
            print('WARNING: E_nu1*M_nu1<=-6ln(eps/2)', mu, nu1, nu2)
        if E_nu2 * M_nu2 < -6 * log(eps / 2):
            print('WARNING: E_nu2*M_nu2<-6ln(eps/2)', mu, nu1, nu2)
        if Y1_l_ZZRM(mu, nu1, nu2, eps, M_mu, M_nu1, M_nu2, N_mu, N_nu1,
                     N_nu2) * N_mu * mu * exp(-mu) < -3 * log(eps / 2):
            print('WARNING: p1*M1_l<-3ln(eps/2)', mu, nu1, nu2)
        return -100500


def R_sec_ZZRM(mu, nu1, nu2, eps, f_ec, R_ver, E_mu, E_nu1, E_nu2, M_mu, M_nu1,
               M_nu2, N_mu, N_nu1, N_nu2):
    return R_ver * r_pa_ZZRM(mu, nu1, nu2, eps, f_ec, E_mu, E_nu1, E_nu2, M_mu,
                             M_nu1, M_nu2, N_mu, N_nu1, N_nu2)
