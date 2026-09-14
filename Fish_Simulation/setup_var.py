from .params import Params
import numpy as np

def setup_var(tf, x_init, y_init, t_pause):
    """Port of setupVar.m"""
    p = Params()

    # Time Variables
    p = get_time_vars(p, tf)

    # System
    p = get_sys_vars(p, x_init, y_init)  # Number of elements+inputs/Init conditions

    # Fish
    p = get_fish_vars(p)  # Fish shape and Hydro Force Constant

    # Control
    p = get_control_vars(p)  # Control Variables

    # Plot
    p.t_pause = t_pause

    return p

def get_time_vars(p, tf):
    """Port of getTimeVars.m"""
    p.dt = 0.01
    p.tf = tf
    p.nt = int(round(tf / p.dt)) + 1
    p.t = np.linspace(0.0, p.tf, p.nt)  # equivalent to MATLAB 0:dt:tf
    return p


def get_sys_vars(p, x_init, y_init):
    """Port of getSysVars.m"""
    # Model Variables
    p.n = 30          # number of elements
    p.nn = p.n + 1      # number of nodes
    p.nin = 3          # number of inputs
    p.ndof = 3 * p.nn    # total number of dof
    p.ns = 2 * p.ndof

    # Initial Conditions
    p.xInit = x_init    # initial x position
    p.yInit = y_init    # initial y position
    p.thInit = 0  # initial th (deg)
    p.vxInit = 0.0        # initial x velocity of all nodes
    p.vyInit = 0.0        # initial y velocity of all nodes
    p.wInit = 0.0         # initial rotational velocity

    p.x0 = np.array(
        [p.xInit, p.yInit, p.thInit, p.vxInit, p.vyInit, p.wInit],
        dtype=float,
    )
    return p


def get_fish_vars(p):
    """Port of getFishVars.m"""
    L = 1.0             # assuming length is 1m
    rbaseline = 0.05     # assuming maximum height is 0.1m
    # left is tail, right is head
    rvec = np.linspace(1.0, 1.0, p.n) * rbaseline

    SAvec = 2 * np.pi * rvec * L / p.n
    rouL = 1.0e3          # density of water
    cDrag = 0.167         # drag coefficient
    uwvec = 0.5 * cDrag * SAvec * rouL  # hydroforce drag constant
    uwvec = np.append(uwvec, uwvec[-1])

    p.rvec = rvec
    p.L = L
    p.uwvec = uwvec
    return p

def get_control_vars(p):
    """Port of getControlVars.m"""
    p.A = 1.0
    p.w = 1.5  # frequency of oscillation (baseline 1.5)
    p.AmpCo = 200.0 * np.ones(p.nin) / p.n
    #p.AmpCo = np.linspace(50.0/p.n, 150.0/p.n, p.nin)
    p.offsetCo = 100.0 * np.ones(p.nin) / p.n
    return p
