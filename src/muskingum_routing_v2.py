import numpy as np
from typing import Optional, Dict, Tuple


def muskingum_route_v2(inflow, K, X, dt, n_reaches=1, Q0=None):
    I = np.asarray(inflow, dtype=float)
    n = len(I)
    if n == 0:
        return np.array([])
    
    if Q0 is None:
        Q0 = float(I[0])
    
    X = float(np.clip(X, 0.0, 0.5))
    K = float(max(K, 1e-12))
    dt = float(max(dt, 1e-12))
    
    denom = (K - K * X + 0.5 * dt)
    if abs(denom) < 1e-12:
        return np.full(n, Q0, dtype=float)
    
    C0 = (-K * X + 0.5 * dt) / denom
    C1 = (K * X + 0.5 * dt) / denom
    C2 = (K - K * X - 0.5 * dt) / denom
    
    C0 = np.clip(C0, 0.0, 1.0)
    C1 = np.clip(C1, 0.0, 1.0)
    C2 = np.clip(C2, 0.0, 1.0)
    C_sum = C0 + C1 + C2
    if C_sum > 0:
        C0 /= C_sum
        C1 /= C_sum
        C2 /= C_sum
    
    C0 = np.clip(C0, 0.0, 1.0)
    C1 = np.clip(C1, 0.0, 1.0)
    C2 = 1.0 - C0 - C1
    C2 = np.clip(C2, 0.0, 1.0)
    
    Q = np.zeros(n)
    Q[0] = Q0
    
    for i in range(1, n):
        Q[i] = C0 * I[i] + C1 * I[i-1] + C2 * Q[i-1]
    
    return np.maximum(Q, 0.0)


def muskingum_cascade(inflow, K=4.877, X=0.145, dt=1.0, n_reaches=5, Q0=None):
    if n_reaches <= 1:
        return muskingum_route_v2(inflow, K, X, dt, 1, Q0)
    
    current_flow = inflow.copy()
    for reach in range(n_reaches):
        current_flow = muskingum_route_v2(
            current_flow, 
            K / n_reaches, 
            X, 
            dt, 
            1, 
            current_flow[0] if len(current_flow) > 0 else Q0
        )
    
    return current_flow


def apply_upstream_routing_v2(upstream_flow, xaj_model_flow, K=4.877, X=0.145, n_reaches=5, dt=1.0):
    if upstream_flow is None or len(upstream_flow) == 0:
        return xaj_model_flow
    
    routed_upstream = muskingum_cascade(upstream_flow, K=K, X=X, n_reaches=n_reaches, dt=dt)
    result = xaj_model_flow + routed_upstream
    return result


DEFAULT_V2_PARAMS = {
    "B": 0.3,
    "C": 0.2,
    "WM": 150.0,
    "WUM": 23.867202119765466,
    "WLM": 60.0,
    "IM": 0.02,
    "SM": 30.921720241298537,
    "EX": 1.1236984984316019,
    "K": 1.2,
    "KG": 0.3697662256735042,
    "KI": 0.2,
    "CG": 0.9977960537453832,
    "CI": 0.85,
    "CS": 0.7232327466183337,
    "L": 1,
    "X": 0.2696357296913911,
    "K_res": 4.877223153101952,
    "X_res": 0.14468991188350286,
    "n": 5
}

V2_PARAM_BOUNDS = {
    'B': (0.2, 0.4),
    'C': (0.1, 0.3),
    'WM': (100.0, 200.0),
    'WUM': (15.0, 35.0),
    'WLM': (40.0, 80.0),
    'IM': (0.0, 0.05),
    'SM': (20.0, 45.0),
    'EX': (0.8, 1.5),
    'K': (0.8, 1.5),
    'KG': (0.25, 0.5),
    'KI': (0.1, 0.3),
    'CG': (0.99, 0.999),
    'CI': (0.6, 0.95),
    'CS': (0.5, 0.9),
    'L': (0, 5),
    'X': (0.15, 0.4),
    'K_res': (3.0, 7.0),
    'X_res': (0.1, 0.2),
    'n': (1, 10)
}