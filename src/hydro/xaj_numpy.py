# -*- coding: utf-8 -*-
"""
新安江模型 (XAJ)
独立实现，不依赖项目其他模块
"""
import numpy as np
from typing import Dict

XAJ_PARAM_BOUNDS = {
    'K': (0.7, 1.3),
    'B': (0.1, 0.5),
    'IM': (0.001, 0.1),
    'WUM': (10.0, 60.0),   # 对应原 um
    'WLM': (50.0, 150.0),  # 对应原 lm
    'WM': (100.0, 330.0),  # 对应原 dm，新版改为率定总蓄水容量 WM
    'C': (0.1, 0.5),
    'SM': (10.0, 80.0),
    'EX': (1.0, 2.0),
    'KI': (0.1, 0.5),
    'KG': (0.1, 0.5),
    'CS': (0.5, 0.98),
    'L': (0, 20),
    'CI': (0.5, 0.98),
    'CG': (0.98, 0.999),
}

XAJ_PARAM_ORDER = [
    'k', 'b', 'im', 'um', 'lm', 'dm', 'c',
    'sm', 'ex', 'ki', 'kg', 'cs', 'l', 'ci', 'cg'
]


def _constrain_ki_kg(ki: float, kg: float) -> tuple:
    if ki + kg >= 1.0:
        total = ki + kg
        ki = ki / total * 0.99
        kg = kg / total * 0.99
    return ki, kg


import math
import numpy as np
from typing import Dict

def run_xaj_model(
    precip: np.ndarray,
    evap: np.ndarray,
    params: Dict[str, float],
    area: float = 584.0,
) -> np.ndarray:
    """
    运行新安江模型 (基于 xaj_core 核心代码完全翻译的纯 NumPy 版本)
    
    Args:
        precip: 降水序列 (mm)
        evap: 蒸发序列 (mm)
        params: 参数字典
        area: 流域面积 (km²)
        
    Returns:
        模拟流量序列 (m³/s)
    """
    n = len(precip)
    if n == 0:
        return np.array([])
        
    # --- 参数提取与默认值设置 (完全对齐 xaj_core) ---
    K = params.get('K', 1.1)
    B = params.get('B', 0.3)
    C = params.get('C', 0.15)
    WM = params.get('WM', 120.0)
    WUM = params.get('WUM', 20.0)
    WLM = params.get('WLM', 70.0)
    IM = params.get('IM', 0.01)
    SM = params.get('SM', 50.0)
    EX = params.get('EX', 1.5)
    
    KG = params.get('KG', 0.3)
    KI = params.get('KI', 0.15)
    CG = params.get('CG', 0.95)
    CI = params.get('CI', 0.8)
    CS = params.get('CS', 0.6)
    
    L = int(params.get('L', 1))
    X = params.get('X', 0.3)
    T = params.get('T', 1.0)
    
    # 初始状态参数
    WUM_init = params.get('WUM_init', 10.0)
    WLM_init = params.get('WLM_init', 30.0)
    WDM_init = params.get('WDM_init', 20.0)
    S1_init = params.get('S1', 10.0)
    FR1 = params.get('FR1', 0.3)
    Q_init = params.get('Q', 5.0)
    
    n_reaches = int(params.get('n', 1))
    
    # 派生参数
    WDM = max(0.0, float(WM - WUM - WLM))
    WDM_init = min(WDM_init, WDM)
    WMM = float(WM)
    SMM = float(SM)
    Sm = float(SM)

    # --- 辅助边界函数 ---
    def _clip01(x, eps=1e-12):
        return float(np.clip(x, 0.0, 1.0 - eps))
    def _pos(x, eps=1e-12):
        return float(max(x, eps))

    # --- 状态与中间序列初始化 ---
    WU = np.zeros(n); WL = np.zeros(n); WD = np.zeros(n)
    EU = np.zeros(n); EL = np.zeros(n); ED = np.zeros(n)
    E = np.zeros(n); PE = np.zeros(n); W = np.zeros(n)
    R = np.zeros(n); FR = np.zeros(n); S1 = np.zeros(n)
    RS = np.zeros(n); RI = np.zeros(n); RG = np.zeros(n)
    QS = np.zeros(n); QI = np.zeros(n); QG = np.zeros(n)
    QT = np.zeros(n); Qt = np.zeros(n)
    
    Q_reaches = np.zeros((n_reaches + 1, n))
    
    P_perv = (1.0 - IM) * precip
    P_im = IM * precip
    EP = evap * K
    U = area / (3.6 * T)

    # 初始化随时间迭代的状态变量
    wu = WUM_init
    wl = WLM_init
    wd = WDM_init
    s1 = S1_init
    fr_prev = FR1

    # --- 核心逐时段演算 ---
    for i in range(n):
        # 1. 土壤含水量计算 (WU, WL, WD)
        if i == 0:
            wu = WUM_init
            wl = WLM_init
            wd = WDM_init
        else:
            infi = PE[i - 1] - R[i - 1]
            wu = wu + infi
            if wu < 0:
                wl = wl + wu
                wu = 0
                if wl < 0:
                    wd = wd + wl
                    wl = 0
                    if wd < 0:
                        wd = 0
            if wu > WUM:
                wl = wl + wu - WUM
                wu = WUM
                if wl > WLM:
                    wd = wd + wl - WLM
                    wl = WLM
                    if wd > WDM:
                        wd = WDM
        WU[i] = wu; WL[i] = wl; WD[i] = wd

        # 2. 蒸散发计算 (EU, EL, ED)
        Pp = P_perv[i]
        ep = EP[i]
        if wu + Pp >= ep:
            eu = ep; el = 0; ed = 0
        elif wu + Pp < ep and wl >= C * WLM:
            eu = wu + Pp
            el = (ep - eu) * (wl / _pos(WLM))
            ed = 0
        elif wu + Pp < ep and C * (ep - eu) <= wl and wl < C * WLM:
            eu = wu + Pp
            el = C * (ep - eu)
            ed = 0
        elif wu + Pp < ep and wl < C * (ep - eu):
            eu = wu + Pp
            el = wl
            ed = C * (ep - eu) - el
        EU[i] = eu; EL[i] = el; ED[i] = ed

        # 3. 总蒸散发与净雨 (E, PE, W)
        E[i] = eu + el + ed
        PE[i] = Pp - E[i]
        W[i] = wu + wl + wd

        # 4. 产流计算 (R)
        if PE[i] > 0:
            one_minus_W_over_WM = _clip01(1.0 - W[i] / _pos(WM))
            a = WMM * (1.0 - math.pow(one_minus_W_over_WM, 1.0 / (1.0 + B)))
            if a + PE[i] <= WMM:
                inner = _clip01(1.0 - (PE[i] + a) / _pos(WMM))
                R[i] = PE[i] + W[i] - WM + WM * math.pow(inner, (B + 1.0))
            else:
                R[i] = PE[i] - (WM - W[i])
        else:
            R[i] = 0.0

        # 5. 产流面积比重计算 (FR)
        if R[i] > 0:
            denom = _pos(PE[i])
            fr = R[i] / denom
            FR[i] = min(1.0, max(fr, 1e-9))
        else:
            FR[i] = fr_prev if i > 0 else FR1

        # 6. 分水源计算 (RS, RG, RI)
        S1[i] = s1
        FRi = max(float(FR[i]), 1e-9)
        FRim1 = max(float(FR[i-1]) if i > 0 else FR1, 1e-9)

        if PE[i] > 0:
            if i == 0:
                ratio = ((S1[i]*FR1) / _pos(FRi)) / _pos(Sm)
            else:
                ratio = ((S1[i]*FRim1) / _pos(FRi)) / _pos(Sm)
            ratio = _clip01(ratio)
            AU = SMM * (1.0 - math.pow(1.0 - ratio, 1.0 / (1.0 + EX)))

            if PE[i] + AU < SMM:
                if i == 0:
                    base = PE[i] + (S1[i]*FR1)/_pos(FRi) - Sm
                else:
                    base = PE[i] + (S1[i]*FRim1)/_pos(FRi) - Sm
                inner = _clip01(1.0 - (PE[i] + AU) / _pos(SMM))
                RS_raw = FRi * (base + Sm * math.pow(inner, 1.0 + EX))
            else:
                if i == 0:
                    RS_raw = FRi * (PE[i] + (S1[i]*FR1)/_pos(FRi) - Sm)
                else:
                    RS_raw = FRi * (PE[i] + (S1[i]*FRim1)/_pos(FRi) - Sm)

            R_i = float(R[i]) if np.isfinite(R[i]) else 0.0
            RS[i] = float(np.clip(RS_raw, 0.0, R_i))

            if i == 0:
                S = (S1[i]*FR1)/_pos(FRi) + (R[i]-RS[i]) / _pos(FRi)
            else:
                S = (S1[i]*FRim1)/_pos(FRi) + (R[i]-RS[i]) / _pos(FRi)

            RI[i] = KI * S * FRi
            RG[i] = KG * S * FRi
            s1 = S * (1.0 - KI - KG)
        else:
            if i == 0: S = (S1[i]*FR1)/_pos(FRi)
            else:      S = (S1[i]*FRim1)/_pos(FRi)
            s1 = S * (1.0 - KG - KI)
            RS[i] = 0.0
            RG[i] = KG * S * FRi
            RI[i] = KI * S * FRi
            
        fr_prev = FR[i]

        # 7. 坡面及地下水汇流计算 (QS, QI, QG)
        QS[i] = max(0.0, float((RS[i] + P_im[i]) * U))

        if i == 0:
            QI[i] = 1/3 * Q_init
            QG[i] = 1/3 * Q_init
        else:
            QI[i] = CI*QI[i-1] + (1-CI)*RI[i]*U
            QG[i] = CG*QG[i-1] + (1-CG)*RG[i]*U
            
        QT[i] = QS[i] + QI[i] + QG[i]
        
        # 滞后与消退
        if 0 <= i <= L:
            Qt[i] = Q_init
        else:
            Qt[i] = Qt[i-1]*CS + (1-CS)*QT[i-L]

        # 8. 河道马斯京根演算法演算 (Qi)
        Q_reaches[0, i] = Qt[i]
        if n_reaches > 0:
            K_l = T
            x_l = 0.5 - n_reaches*(1-2*X)/2
            denom = 0.5*T + K_l - K_l*x_l
            C0 = (0.5*T - K_l*x_l) / denom
            C1 = (0.5*T + K_l*x_l) / denom
            C2 = 1.0 - C0 - C1
            
            for p in range(n_reaches):
                I2 = Q_reaches[p, i]
                if i == 0:
                    I1 = Q_init
                    Q1_prev = Q_init
                else:
                    I1 = Q_reaches[p, i-1]
                    Q1_prev = Q_reaches[p+1, i-1]
                Q_reaches[p+1, i] = C0*I2 + C1*I1 + C2*Q1_prev

    # 提取最终断面的演进流量序列
    Q_final = Q_reaches[-1, :]
    
    # 清理所有潜在的负值以符合自然规律
    return np.clip(np.nan_to_num(Q_final, nan=0.0), 0.0, None)