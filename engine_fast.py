"""
engine_fast.py — Skompilowane (Numba @njit) pętle Eulera silnika symulacji.

Pętle forward/backward pass są rekurencyjne (v[i+1] zależy od v[i]),
więc nie da się ich zwektoryzować numpy. Numba kompiluje je do kodu
maszynowego (LLVM), dając ~20-50× przyspieszenie względem czystego Pythona.

Funkcje przyjmują parametry jako pojedyncze floaty (nie obiekt Parameters),
bo Numba nie obsługuje dataclass/@property. Profil pochyleń przekazywany
jest jako trzy tablice numpy (starts, ends, grads).

Pierwsze wywołanie każdej funkcji jest wolne (kompilacja JIT ~kilka sekund),
kolejne są błyskawiczne.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import numpy as np
from numba import njit

G_CONST = 9.81


@njit(cache=True, fastmath=True)
def gradient_at_njit(x, starts, ends, grads):
    """Odczyt pochylenia [‰] w punkcie x z profilu (tablice segmentów)."""
    TOL = 1.0
    n = len(starts)
    for k in range(n):
        if starts[k] - TOL <= x <= ends[k] + TOL:
            return grads[k]
    # fallback - skrajny segment
    if x < starts[0]:
        return grads[0]
    return grads[n - 1]


@njit(cache=True, fastmath=True)
def a_ham_njit(v_ms, mu_b_base, braked_frac):
    """
    Opóźnienie hamowania = sufit przyczepności wg TSI 4.2.4.6.1 [m/s²].

    μ_b(v): stałe do 250 km/h, spadek liniowy o 0,05 do 350 km/h, zamrożone
    powyżej. a_ham(v) = μ_b(v)·braked_frac·g — jazda po suficie przyczepności.
    """
    v_kmh = v_ms * 3.6
    if v_kmh <= 250.0:
        mu = mu_b_base
    else:
        vv = v_kmh if v_kmh < 350.0 else 350.0
        mu = mu_b_base - 0.05 * (vv - 250.0) / 100.0
    return mu * braked_frac * G_CONST


@njit(cache=True, fastmath=True)
def forward_pass_njit(
    L,
    dx,
    v_max,
    m_eff,
    F_max,
    P_eff_max,
    v_breakpoint,
    davis_A,
    davis_B,
    davis_C,
    dx_coast,
    m,
    starts,
    ends,
    grads,
):
    """
    Forward pass: rozpędzanie → jazda ustalona → coasting.

    Zwraca (x, v, phase) jako tablice numpy.
    Logika identyczna z simulation.forward_pass, ale skompilowana.
    """
    N = int(np.ceil(L / dx)) + 1
    x = np.arange(N) * dx
    v = np.zeros(N)
    phase = np.zeros(N, dtype=np.int8)

    x_start_coast = L - dx_coast

    v[0] = 0.0
    phase[0] = 1

    for i in range(N - 1):
        # Decyzja o fazie
        if x[i] >= x_start_coast:
            current_phase = 3
        elif v[i] < v_max - 0.01:
            current_phase = 1
        else:
            current_phase = 2

        phase[i] = current_phase

        # Wypadkowa siła sterująca u(v, x)
        if current_phase == 1:
            # F_traction(v)
            if v[i] <= v_breakpoint:
                u = F_max
            else:
                u = P_eff_max / v[i]
        elif current_phase == 2:
            # napęd kompensuje opory + grawitację
            F_op_loc = davis_A + davis_B * v[i] + davis_C * v[i] * v[i]
            i_prom = gradient_at_njit(x[i], starts, ends, grads)
            F_g_loc = m * G_CONST * i_prom / 1000.0
            u = F_op_loc + F_g_loc
        else:  # faza 3, coasting
            u = 0.0

        # Opory i grawitacja
        F_op = davis_A + davis_B * v[i] + davis_C * v[i] * v[i]
        i_prom = gradient_at_njit(x[i], starts, ends, grads)
        F_g = m * G_CONST * i_prom / 1000.0

        F_net = u - F_op - F_g

        # Faza 2: trzymanie v_max
        if current_phase == 2:
            v[i + 1] = v_max
            phase[i + 1] = 2
            continue

        # Euler w dziedzinie drogi
        v_squared_next = v[i] * v[i] + 2.0 * F_net / m_eff * dx

        if v_squared_next <= 0.0:
            v[i + 1] = 0.0
            for j in range(i + 1, N):
                v[j] = 0.0
                phase[j] = current_phase
            break

        v[i + 1] = np.sqrt(v_squared_next)

        if current_phase == 1 and v[i + 1] > v_max:
            v[i + 1] = v_max

    # Ostatni indeks
    if x[N - 1] >= x_start_coast:
        phase[N - 1] = 3
    elif v[N - 1] >= v_max - 0.01:
        phase[N - 1] = 2
    else:
        phase[N - 1] = 1

    return x, v, phase


@njit(cache=True, fastmath=True)
def backward_pass_njit(
    L,
    dx,
    m_eff,
    mu_b_base,
    braked_frac,
    davis_A,
    davis_B,
    davis_C,
    m,
    starts,
    ends,
    grads,
):
    """
    Backward pass: hamowanie wstecz od x=L (v=0).

    Zwraca tablicę v_bwd. Logika identyczna z simulation.backward_pass.
    """
    N = int(np.ceil(L / dx)) + 1
    x = np.arange(N) * dx
    v_bwd = np.zeros(N)
    v_bwd[N - 1] = 0.0

    for i in range(N - 1, 0, -1):
        v_current = v_bwd[i]

        # Opóźnienie wg sufitu przyczepności TSI 4.2.4.6.1 (zależne od v)
        a_dec = a_ham_njit(v_current, mu_b_base, braked_frac)

        # F_brake_required dla a_dec(v)
        F_op = davis_A + davis_B * v_current + davis_C * v_current * v_current
        i_prom = gradient_at_njit(x[i], starts, ends, grads)
        F_g = m * G_CONST * i_prom / 1000.0

        F_brake = m_eff * a_dec - F_op - F_g
        if F_brake < 0.0:
            F_brake = 0.0

        F_net_decel = F_brake + F_op + F_g

        v_squared_prev = v_current * v_current + 2.0 * F_net_decel / m_eff * dx
        if v_squared_prev <= 0.0:
            v_bwd[i - 1] = 0.0
        else:
            v_bwd[i - 1] = np.sqrt(v_squared_prev)

    return v_bwd
