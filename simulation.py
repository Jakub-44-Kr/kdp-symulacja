"""
simulation.py — Silnik symulacji ruchu pociągu KDP.

Implementacja całkowania równania ruchu (29) z pracy metodą Eulera
w dziedzinie drogi, z dwoma przebiegami:
  - forward pass: rozpędzanie → jazda ustalona → coasting (od x=0)
  - backward pass: hamowanie (od x=L, wstecz)
  - meeting point: złożenie obu krzywych w finalny profil v(x)

Wszystkie wartości w SI.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from parameters import Parameters
from physics import (
    F_brake_required,
    F_davis,
    F_gravity,
    F_resultant_in_phase,
    TrackProfile,
)

# ═══════════════════════════════════════════════════════════════════════════
#  KONTENERY DANYCH
# ═══════════════════════════════════════════════════════════════════════════


# Stała przyspieszenia ziemskiego (lokalna kopia dla wektoryzacji)
G_CONST: float = 9.81


# ═══════════════════════════════════════════════════════════════════════════
#  WEKTOROWE WERSJE FUNKCJI FIZYCZNYCH (post-processing)
# ═══════════════════════════════════════════════════════════════════════════


def _F_traction_vec(v: np.ndarray, p: Parameters) -> np.ndarray:
    """Wektorowa F_traction: F_max do v_b, potem P/v."""
    F = np.full_like(v, p.F_max)
    mask_power = v > p.v_breakpoint
    F[mask_power] = p.P_eff_max / v[mask_power]
    return F


def _gradient_profile_vec(x: np.ndarray, profile: TrackProfile) -> np.ndarray:
    """
    Wektorowy odczyt pochylenia [‰] dla każdego punktu x.

    Dla każdego segmentu (x_start, x_end, i) ustawia i tam gdzie x w zakresie.
    Tolerancja na końcach dla zaokrągleń float.
    """
    grad = np.zeros_like(x)
    TOL = 1.0
    for x_start, x_end, i_promille in profile:
        mask = (x >= x_start - TOL) & (x <= x_end + TOL)
        grad[mask] = i_promille
    return grad


def _F_brake_required_vec(
    v: np.ndarray, gradient_local: np.ndarray, p: Parameters
) -> np.ndarray:
    """
    Wektorowa F_brake_required dla zadanego opóźnienia a_brake_max.

    F_req = m_eff·a_decel - F_op(v) - F_grav  (≥ 0)
    """
    F_op = p.davis_A + p.davis_B * v + p.davis_C * v * v
    F_grav = p.m * G_CONST * gradient_local / 1000.0
    F_req = p.m_eff * p.a_brake_max - F_op - F_grav
    return np.maximum(0.0, F_req)


@dataclass
class SimulationProfile:
    """
    Surowy profil dynamiczny pociągu wzdłuż trasy.

    Wszystkie tablice mają tę samą długość N = len(x).
    Indeks 0 odpowiada x=0 (start), ostatni indeks N-1 odpowiada x=L.

    Atrybuty:
        x:        Pozycja [m]
        v:        Prędkość [m/s]
        t:        Czas [s] (akumulowany)
        a:        Przyspieszenie [m/s²]
        phase:    Numer fazy (1/2/3/4)
        F_tr:     Siła trakcyjna [N] (≥0)
        F_brake:  Siła hamulcowa CAŁKOWITA [N] (≥0)
        F_op:     Opór ruchu wg Davisa [N]
        F_grav:   Składowa grawitacyjna [N] (+ dla wzniesienia)
        gradient: Pochylenie lokalne [‰]
    """

    x: np.ndarray
    v: np.ndarray
    t: np.ndarray
    a: np.ndarray
    phase: np.ndarray
    F_tr: np.ndarray
    F_brake: np.ndarray
    F_op: np.ndarray
    F_grav: np.ndarray
    gradient: np.ndarray

    # Punkty przełączania faz
    x1: float = 0.0  # koniec rozpędzania (osiągnięcie v_set)
    x2: float = 0.0  # początek coastingu
    x3: float = 0.0  # początek hamowania (meeting point)

    # Czy pociąg osiągnął v_set?
    reached_v_set: bool = False

    @property
    def T_total(self) -> float:
        """Całkowity czas przejazdu [s]."""
        return float(self.t[-1])

    @property
    def v_avg(self) -> float:
        """Średnia prędkość [m/s]."""
        return float(self.x[-1] / self.t[-1])

    @property
    def v_max_reached(self) -> float:
        """Maksymalna osiągnięta prędkość [m/s]."""
        return float(np.max(self.v))


# ═══════════════════════════════════════════════════════════════════════════
#  FORWARD PASS — rozpędzanie, jazda ustalona, coasting
# ═══════════════════════════════════════════════════════════════════════════


def forward_pass(
    p: Parameters, profile: TrackProfile
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Całkowanie ruchu do przodu metodą Eulera w dziedzinie drogi.

    Fazy w forward pass:
      1) Rozpędzanie: F_tr(v), aż do v_set
      2) Jazda ustalona: utrzymanie v_set, F_tr = F_op + F_grav
      3) Coasting: F_tr = 0, pociąg toczy się pod wpływem oporów

    Zatrzymujemy się gdy x ≥ L lub gdy v ≤ v_min_num (zabezpieczenie awaryjne).

    Args:
        p: Parametry symulacji.
        profile: Profil trasy.

    Returns:
        Krotka (x, v, phase) - tablice numpy o tej samej długości.
    """
    N = int(math.ceil(p.L / p.dx)) + 1
    x = np.arange(N, dtype=np.float64) * p.dx
    v = np.zeros(N, dtype=np.float64)
    phase = np.zeros(N, dtype=np.int8)

    # Punkt rozpoczęcia coastingu (mierzony od końca trasy)
    x_start_coast = p.L - p.dx_coast

    # Punkt startu - faza 1 (rozpędzanie)
    v[0] = 0.0
    phase[0] = 1

    for i in range(N - 1):
        # Decyzja o fazie na podstawie stanu (v, x)
        if x[i] >= x_start_coast:
            current_phase = 3  # coasting
        elif v[i] < p.v_max - 0.01:  # tolerancja 1 cm/s
            current_phase = 1  # rozpędzanie
        else:
            current_phase = 2  # jazda ustalona

        phase[i] = current_phase

        # Wypadkowa siła sterująca
        u = F_resultant_in_phase(current_phase, v[i], x[i], p, profile)

        # Opory i grawitacja (zawsze obecne, niezależnie od fazy)
        F_op = F_davis(v[i], p)
        F_g = F_gravity(x[i], p, profile)

        # Wypadkowa netto
        F_net = u - F_op - F_g

        # Specjalna obsługa fazy 2 (jazda ustalona) - utrzymanie v_set
        if current_phase == 2:
            v[i + 1] = p.v_max
            phase[i + 1] = 2
            continue

        # Całkowanie Eulera w dziedzinie drogi: v²_{i+1} = v²_i + 2·F_net/m_eff · Δx
        v_squared_next = v[i] ** 2 + 2.0 * F_net / p.m_eff * p.dx

        # Zabezpieczenie: jeśli wyszłoby v² < 0, zatrzymujemy (awaryjne)
        if v_squared_next <= 0:
            v[i + 1] = 0.0
            # Wypełnij resztę tablicy zerami i kończymy
            v[i + 1 :] = 0.0
            phase[i + 1 :] = current_phase
            break

        v[i + 1] = math.sqrt(v_squared_next)

        # Ograniczenie: w fazie 1 nie przekraczamy v_max
        if current_phase == 1 and v[i + 1] > p.v_max:
            v[i + 1] = p.v_max

    # Ostatni indeks (faza wynika z położenia)
    if x[-1] >= x_start_coast:
        phase[-1] = 3
    elif v[-1] >= p.v_max - 0.01:
        phase[-1] = 2
    else:
        phase[-1] = 1

    return x, v, phase


# ═══════════════════════════════════════════════════════════════════════════
#  BACKWARD PASS — hamowanie wstecz od x=L
# ═══════════════════════════════════════════════════════════════════════════


def backward_pass(p: Parameters, profile: TrackProfile) -> np.ndarray:
    """
    Całkowanie ruchu wstecz od końca trasy.

    Symulujemy hamowanie z opóźnieniem a_brake_max, zaczynając od v=0 w x=L.
    Idziemy w stronę x=0 i w każdym kroku obliczamy jaką prędkość musiał
    mieć pociąg żeby zatrzymać się w x=L.

    Równanie wsteczne (z m_eff · v · dv/dx = -F_brake - F_op - F_grav):
        v²_{i-1} = v²_i + 2 · (F_brake + F_op - F_grav_wsteczne) / m_eff · Δx

    UWAGA - znak F_grav: gdy idziemy WSTECZ, składowa grawitacyjna działa
    odwrotnie do kierunku ruchu numerycznego, więc znak ZMIENIAMY:
    wzniesienie (i>0) wspomaga hamowanie idąc do przodu = utrudnia "rozpędzanie wstecz"
    co matematycznie oznacza że v_{i-1} jest mniejsze.

    Args:
        p: Parametry symulacji.
        profile: Profil trasy.

    Returns:
        Tablica v_bwd o tej samej długości co forward pass (indeksowana od x=0).
        v_bwd[N-1] = 0 (start hamowania od końca), v_bwd[0] = teoretyczna v
        z której można by zacząć hamować już na początku trasy (zazwyczaj b.duża).
    """
    N = int(math.ceil(p.L / p.dx)) + 1
    x = np.arange(N, dtype=np.float64) * p.dx
    v_bwd = np.zeros(N, dtype=np.float64)

    # Start: v=0 w x=L (ostatni indeks)
    v_bwd[N - 1] = 0.0

    for i in range(N - 1, 0, -1):
        v_current = v_bwd[i]

        # Wymagana siła hamulcowa dla zadanego opóźnienia
        F_brake = F_brake_required(v_current, x[i], p, profile, p.a_brake_max)
        F_op = F_davis(v_current, p)
        F_g = F_gravity(x[i], p, profile)

        # Idąc wstecz, składowa grawitacyjna wspomaga lub przeszkadza hamowaniu
        # ze znakiem PRZECIWNYM niż w forward pass:
        # - wzniesienie (F_g > 0): wspomaga hamowanie = mniejsza v poprzednia
        # - spadek (F_g < 0): przeszkadza hamowaniu = większa v poprzednia
        # Sumarycznie: w wstecz, F_net_decel = F_brake + F_op + F_g
        F_net_decel = F_brake + F_op + F_g

        # Krok wsteczny: v²_{i-1} = v²_i + 2·F_net_decel/m_eff·dx
        v_squared_prev = v_current**2 + 2.0 * F_net_decel / p.m_eff * p.dx

        if v_squared_prev <= 0:
            v_bwd[i - 1] = 0.0
        else:
            v_bwd[i - 1] = math.sqrt(v_squared_prev)

    return v_bwd


# ═══════════════════════════════════════════════════════════════════════════
#  MEETING POINT — punkt rozpoczęcia hamowania
# ═══════════════════════════════════════════════════════════════════════════


def find_meeting_point(v_fwd: np.ndarray, v_bwd: np.ndarray) -> int:
    """
    Znajduje indeks x_3 = pierwszy punkt gdzie krzywa forward przekracza
    (lub równa) krzywą backward.

    Logika:
      - Przed punktem hamowania: v_fwd < v_bwd (możemy jeszcze jechać szybciej
        bo jest dużo miejsca do zahamowania)
      - W punkcie hamowania: v_fwd ≈ v_bwd (musimy zacząć hamować)
      - Po punkcie hamowania: v_bwd < v_fwd (gdybyśmy nie zaczęli hamować,
        nie zdążylibyśmy się zatrzymać)

    Args:
        v_fwd: Profil prędkości z forward pass.
        v_bwd: Profil prędkości z backward pass.

    Returns:
        Indeks tablicy odpowiadający rozpoczęciu hamowania.
        Jeśli krzywe się nie przecinają (pociąg już hamuje od początku),
        zwraca 0. Jeśli nigdy nie hamuje (krótka trasa) - zwraca N-1.
    """
    N = len(v_fwd)
    # Szukamy pierwszego i takiego że v_fwd[i] >= v_bwd[i]
    diff = v_fwd - v_bwd  # ujemne na początku, dodatnie po meeting point
    # Indeksy gdzie diff >= 0
    candidates = np.where(diff >= 0)[0]
    if len(candidates) == 0:
        return N - 1
    return int(candidates[0])


# ═══════════════════════════════════════════════════════════════════════════
#  GŁÓWNA SYMULACJA — orchestracja
# ═══════════════════════════════════════════════════════════════════════════


def run_simulation(
    p: Parameters, profile: TrackProfile | None = None
) -> SimulationProfile:
    """
    Uruchamia pełną symulację przejazdu.

    Etapy:
      1. Forward pass (rozpędzanie + jazda ustalona + coasting)
      2. Backward pass (hamowanie wstecz)
      3. Złożenie: forward dla x < x_3, backward dla x ≥ x_3
      4. Obliczenie t(x) przez całkowanie dt = dx/v
      5. Obliczenie a(x), sił, gradientów

    Args:
        p: Parametry symulacji.
        profile: Profil trasy. Default: jednorodny o gradient z parameters.

    Returns:
        Wypełniony SimulationProfile.
    """
    if profile is None:
        profile = [(0.0, p.L, p.gradient)]

    # 1. Forward pass
    x, v_fwd, phase_fwd = forward_pass(p, profile)
    N = len(x)

    # 2. Backward pass
    v_bwd = backward_pass(p, profile)

    # 3. Meeting point i złożenie
    idx_brake_start = find_meeting_point(v_fwd, v_bwd)
    v = np.where(np.arange(N) < idx_brake_start, v_fwd, v_bwd)

    # Fazy: forward do x_3, potem faza 4 (hamowanie)
    phase = phase_fwd.copy()
    phase[idx_brake_start:] = 4

    # 4. Czas przez całkowanie dt = dx/v (wektorowo, trapezoidalnie)
    v_avg_seg = 0.5 * (v[:-1] + v[1:])  # średnia prędkość na segmencie
    v_avg_seg = np.maximum(v_avg_seg, p.v_min_num)  # zabezpieczenie przed v≈0
    dt_seg = p.dx / v_avg_seg  # czas na każdym segmencie
    t = np.zeros(N, dtype=np.float64)
    t[1:] = np.cumsum(dt_seg)  # akumulacja czasu

    # 5. Przyspieszenie z różnic skończonych: a = v·dv/dx (wektorowo)
    a = np.zeros(N, dtype=np.float64)
    dv = np.diff(v)  # v[i+1] - v[i], długość N-1
    mask_v = v[:-1] > p.v_min_num
    a[:-1] = np.where(mask_v, v[:-1] * dv / p.dx, 0.0)
    a[-1] = 0.0

    # 6. Siły dla każdego punktu (wektorowo)
    # Opory Davisa: A + B·v + C·v²
    F_op = p.davis_A + p.davis_B * v + p.davis_C * v * v

    # Składowa grawitacyjna: dla każdego x odczytujemy pochylenie z profilu
    gradient_local = _gradient_profile_vec(x, profile)  # ‰ dla każdego x
    F_grav = p.m * G_CONST * gradient_local / 1000.0
    gradient_arr = gradient_local.copy()

    # Siła trakcyjna wg faz
    F_tr = np.zeros(N, dtype=np.float64)
    F_brake_total = np.zeros(N, dtype=np.float64)

    # Faza 1 (rozpędzanie): F_traction(v) - dwuczęściowa charakterystyka
    mask1 = phase == 1
    F_tr[mask1] = _F_traction_vec(v[mask1], p)

    # Faza 2 (jazda ustalona): napęd kompensuje opory + grawitację (≥0)
    mask2 = phase == 2
    F_tr[mask2] = np.maximum(0.0, F_op[mask2] + F_grav[mask2])

    # Faza 3 (coasting): F_tr = 0 (już zainicjalizowane zerami)

    # Faza 4 (hamowanie): wymagana siła hamulcowa
    mask4 = phase == 4
    F_brake_total[mask4] = _F_brake_required_vec(v[mask4], gradient_local[mask4], p)

    # Punkty przełączania
    x1 = float(x[np.where(phase >= 2)[0][0]]) if np.any(phase >= 2) else 0.0
    x2 = float(x[np.where(phase >= 3)[0][0]]) if np.any(phase >= 3) else 0.0
    x3 = float(x[idx_brake_start])
    reached = bool(np.any(v_fwd >= p.v_max - 0.1))

    return SimulationProfile(
        x=x,
        v=v,
        t=t,
        a=a,
        phase=phase,
        F_tr=F_tr,
        F_brake=F_brake_total,
        F_op=F_op,
        F_grav=F_grav,
        gradient=gradient_arr,
        x1=x1,
        x2=x2,
        x3=x3,
        reached_v_set=reached,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST — uruchom: python simulation.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = Parameters.base()
    print(p.summary())
    print()

    print(">>> Uruchamiam symulację...")
    sim = run_simulation(p)

    print()
    print("=== Wyniki symulacji ===")
    print(f"  Czas przejazdu T   = {sim.T_total / 60:.2f} min ({sim.T_total:.0f} s)")
    print(f"  Średnia prędkość   = {sim.v_avg * 3.6:.1f} km/h")
    print(f"  v_max osiągnięte   = {sim.v_max_reached * 3.6:.1f} km/h")
    print(f"  Osiągnięto v_set?  = {sim.reached_v_set}")
    print()
    print("  Punkty przełączania faz:")
    print(f"    x1 (koniec rozpędzania) = {sim.x1 / 1000:.2f} km")
    print(f"    x2 (start coastingu)    = {sim.x2 / 1000:.2f} km")
    print(f"    x3 (start hamowania)    = {sim.x3 / 1000:.2f} km")
    print()
    print("  Pierwsze 5 i ostatnie 5 punktów profilu v(x):")
    print(f"  {'x [km]':>8} {'v [km/h]':>10} {'a [m/s²]':>10} {'faza':>6}")
    for i in list(range(5)) + list(range(len(sim.x) - 5, len(sim.x))):
        print(
            f"  {sim.x[i] / 1000:>8.3f} {sim.v[i] * 3.6:>10.2f} {sim.a[i]:>10.3f} {sim.phase[i]:>6}"
        )
