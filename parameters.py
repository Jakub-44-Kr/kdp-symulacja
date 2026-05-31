"""
parameters.py — Parametry modelu symulacyjnego pociągu KDP.

Plik centralny zawierający wszystkie parametry fizyczne, operacyjne
i numeryczne modelu. Pozostałe moduły importują parametry stąd.

Konwencja jednostek: WSZYSTKO w SI (kg, m, s, N, W, V, A).
Pochylenie wewnętrznie przechowywane jest w promilach [‰] - konwersja
na sin(α) wykonywana jest w physics.py wg wzoru (24).

Odwołania do wzorów odnoszą się do pracy magisterskiej.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
#  🔧 STREFA EDYCJI — SCENARIUSZ BAZOWY (Tabela 4, rozdz. 5.3)
#  Te wartości zmieniasz przy analizie wrażliwości.
# ═══════════════════════════════════════════════════════════════════════════

# --- Parametry ruchu i taboru (objęte analizą wrażliwości OAT) ---
V_MAX_KMH: float = 320.0  # Prędkość eksploatacyjna [km/h] — baza 320, OAT 250-400
MASS_TON: float = 600.0  # Masa składu [t] — baza 600, OAT 450/600/750
POWER_MW: float = 9.0  # Moc znamionowa [MW] — baza 9, OAT 6/9/12
GRADIENT_PROMILE: float = 0.0  # Pochylenie trasy [‰] — baza 0, OAT 0-5
LENGTH_KM: float = 180.0  # Długość odcinka [km] — baza 180 (Wrocław-Poznań)

# --- System zasilania ---
# "AC" = 2x25 kV AC | "DC" = 3 kV DC
POWER_SYSTEM: str = "DC"

# --- Parametry sterownicze (stałe w OAT, mogą stać się zmiennymi) ---
COASTING_DISTANCE_KM: float = (
    5.0  # Δx_coast — dystans początku wybiegu od końca trasy [km]
)


# ═══════════════════════════════════════════════════════════════════════════
#  📐 STAŁE FIZYCZNE I CHARAKTERYSTYKI TABORU
# ═══════════════════════════════════════════════════════════════════════════

# --- Fizyka uniwersalna ---
G: float = 9.81  # Przyspieszenie ziemskie [m/s²]
ROTATING_MASS_FACTOR: float = 0.08  # m_rot = ROTATING_MASS_FACTOR * m  [-]
# Typowo 6-10% dla KDP (Steimel 2008, Lukaszewicz 2009)

# --- Współczynniki oporu Davisa F_op = A + B*v + C*v² ---
# Wartości referencyjne dla KDP (ICE 3 / Velaro, Rochard & Schmid 2000)
# UWAGA: w naszym modelu A,B,C są stałe dla wszystkich klas masy
# (uproszczenie omówione w rozdz. 5.3, przypis 1)
DAVIS_A: float = 2200.0  # [N]      — opór toczny + łożyskowy
DAVIS_B: float = 130.0  # [N·s/m]  — opór mechaniczny zależny od v
DAVIS_C: float = 6.45  # [N·s²/m²] — opór aerodynamiczny

# --- Sprawności (rozdz. 3.5, 3.6) ---
# η_tr różnicowane między systemami zasilania:
# - AC 2×25 kV: 0.88 - niskie prądy (~400 A), małe straty I²R w sieci trakcyjnej
# - DC 3 kV: 0.83 - wysokie prądy (~4000 A), znaczące straty I²R (10× wyższe niż AC)
# Wartości zgodne z literaturą (Steimel 2008, Lukaszewicz 2009, RailEnergy).
ETA_TR_AC: float = 0.88       # Tor przekazywania mocy dla AC
ETA_TR_DC: float = 0.83       # Tor przekazywania mocy dla DC
ETA_TR: float = 0.88          # Domyślne (zachowane dla kompatybilności, używane dla AC)
ETA_REC: float = (
    0.85  # Tor rekuperacji (silnik w trybie generatorowym + przekształtniki)
)
ETA_GRID_AC: float = 1.0  # Receptywność sieci 2x25 kV AC (rozdz. 3.5)
ETA_GRID_DC: float = 0.0  # Receptywność sieci 3 kV DC (pojedynczy pociąg) (rozdz. 3.5)

# --- Moc potrzeb własnych P_aux (norma CLC/TS 50591:2013) ---
P_AUX_KW: float = 250.0  # [kW] — typowa wartość dla KDP (HVAC, oświetlenie, etc.)

# --- Ograniczenia mocy wg systemu zasilania (rozdz. 5.3, wzór 33) ---
P_MAX_AC_MW: float = 12.0  # Górne ograniczenie mocy dla 2x25 kV AC
P_MAX_DC_MW: float = 6.0  # Górne ograniczenie mocy dla 3 kV DC

# --- Napięcia sieci trakcyjnej (do wyliczania prądu pantografu) ---
U_GRID_AC: float = 25_000.0  # [V] — strona pierwotna 2x25 kV (do obciążalności)
U_GRID_DC: float = 3_000.0  # [V] — 3 kV DC

# --- Limity prądu pantografu (TSI ENE, do walidacji) ---
I_MAX_AC: float = 400.0  # [A] — limit ciągły dla 2x25 kV AC
I_MAX_DC: float = 4_000.0  # [A] — limit ciągły dla 3 kV DC

# --- Charakterystyka hamowania (rozdz. 4.2) ---
A_BRAKE_MAX: float = (
    1.2  # [m/s²] — maksymalne opóźnienie hamowania (komfort, Steimel 2008)
)
V_BRAKE_MIN_KMH: float = (
    10.0  # [km/h] — poniżej tej prędkości tylko hamulec mechaniczny
)

# --- Charakterystyka trakcyjna — limity rozruchu ---
A_LAUNCH_MAX: float = 0.7  # [m/s²] — sztywne ograniczenie a podczas rozpędzania
# (rozdz. 5.3, ze względu na adhezję i komfort)
MU_ADHESION: float = 0.30  # Współczynnik przyczepności sucho (informacyjnie)
ADHESION_MASS_FACTOR: float = (
    0.50  # m_ad / m — udział masy na osie napędne (połowa dla 8-wagonowca)
)


# ═══════════════════════════════════════════════════════════════════════════
#  ⚙️ STAŁE NUMERYCZNE
# ═══════════════════════════════════════════════════════════════════════════

DX_STEP: float = 1.0  # [m] — krok przestrzenny całkowania Eulera
# 1 m → dla L=180 km mamy 180 000 kroków, ~sekunda obliczeń

V_MIN_NUMERICAL: float = 0.1  # [m/s] — minimalna prędkość obsługi numerycznej
# (transformacja dt = dx/v wymaga v > 0, rozdz. 4.5)

# Ścieżki I/O
PROJECT_ROOT: Path = Path(__file__).parent
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  🎯 KLASA PARAMETRÓW — używana w całym kodzie
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Parameters:
    """
    Komplet parametrów symulacji w jednostkach SI.

    Klasa frozen — niemodyfikowalna po utworzeniu. Aby zmienić parametr
    w analizie wrażliwości, użyj metody .with_changes() która zwraca KOPIĘ.

    Example:
        >>> p = Parameters.base()
        >>> p_heavy = p.with_changes(m=750_000)  # 750 t zamiast 600 t
    """

    # --- Parametry ruchu i taboru (SI!) ---
    v_max: float  # [m/s] — prędkość eksploatacyjna
    m: float  # [kg]  — masa składu (suchy ciężar)
    P_nom: float  # [W]   — moc znamionowa nominalna
    gradient: float  # [‰]   — pochylenie trasy (jednorodne, ujednolicone na całą L)
    L: float  # [m]   — długość odcinka

    # --- System zasilania ---
    power_system: str  # "AC" lub "DC"

    # --- Parametry sterownicze ---
    dx_coast: float  # [m] — dystans początku wybiegu od końca trasy

    # --- Charakterystyki taboru (dziedziczone z modułu, mogą być nadpisane) ---
    davis_A: float = DAVIS_A
    davis_B: float = DAVIS_B
    davis_C: float = DAVIS_C
    eta_tr: float = ETA_TR
    eta_rec: float = ETA_REC
    P_aux: float = field(default=P_AUX_KW * 1e3)  # [W]
    a_brake_max: float = A_BRAKE_MAX
    v_brake_min: float = field(default=V_BRAKE_MIN_KMH / 3.6)  # [m/s]
    a_launch_max: float = A_LAUNCH_MAX
    rot_mass_factor: float = ROTATING_MASS_FACTOR

    # --- Stałe numeryczne ---
    dx: float = DX_STEP
    v_min_num: float = V_MIN_NUMERICAL

    # ---------- METODY POMOCNICZE ----------

    @property
    def m_eff(self) -> float:
        """Masa efektywna z uwzględnieniem mas wirujących [kg]."""
        return self.m * (1.0 + self.rot_mass_factor)

    @property
    def P_eff_max(self) -> float:
        """
        Efektywna max moc na pantografie wg wzoru (33) z pracy.
        Dla DC ograniczona do 6 MW, dla AC do 12 MW.
        """
        if self.power_system == "DC":
            return min(self.P_nom, P_MAX_DC_MW * 1e6)
        return min(self.P_nom, P_MAX_AC_MW * 1e6)

    @property
    def eta_grid(self) -> float:
        """Receptywność sieci na energię rekuperowaną."""
        return ETA_GRID_DC if self.power_system == "DC" else ETA_GRID_AC

    @property
    def U_grid(self) -> float:
        """Napięcie referencyjne sieci [V] (do liczenia prądu pantografu)."""
        return U_GRID_DC if self.power_system == "DC" else U_GRID_AC

    @property
    def I_grid_limit(self) -> float:
        """Limit prądu pantografu wg TSI ENE [A] (do walidacji)."""
        return I_MAX_DC if self.power_system == "DC" else I_MAX_AC

    @property
    def eta_tr_effective(self) -> float:
        """
        Efektywna sprawność toru przekazywania mocy zależnie od systemu zasilania.

        DC ma niższą sprawność z powodu znacznie wyższych prądów (4 kA vs 400 A)
        i wynikających z tego strat I²R w sieci trakcyjnej (Steimel 2008).
        """
        return ETA_TR_DC if self.power_system == "DC" else ETA_TR_AC

    @property
    def F_max(self) -> float:
        """
        Maksymalna siła trakcyjna [N] - mniejsza z dwóch:
          - ograniczenie adhezji: μ · m_ad · g
          - ograniczenie komfortu: (m + m_rot) · a_launch_max

        Patrz rozdz. 4.2 oraz 5.3 pracy.
        """
        F_adhesion = MU_ADHESION * (ADHESION_MASS_FACTOR * self.m) * G
        F_comfort = self.m_eff * self.a_launch_max
        return min(F_adhesion, F_comfort)

    @property
    def v_breakpoint(self) -> float:
        """
        Prędkość łamania charakterystyki F(v): od F=const do F=P/v [m/s].
        v_b = P_eff_max / F_max
        """
        return self.P_eff_max / self.F_max

    def with_changes(self, **kwargs) -> "Parameters":
        """
        Zwraca KOPIĘ z nadpisanymi wybranymi parametrami.
        Używane do analizy wrażliwości.

        Example:
            >>> p_base = Parameters.base()
            >>> p_dc = p_base.with_changes(power_system="DC")
            >>> p_fast = p_base.with_changes(v_max=400/3.6)
        """
        return replace(self, **kwargs)

    @classmethod
    def base(cls) -> "Parameters":
        """Parametry scenariusza bazowego (Tabela 4)."""
        return cls(
            v_max=V_MAX_KMH / 3.6,
            m=MASS_TON * 1_000.0,
            P_nom=POWER_MW * 1e6,
            gradient=GRADIENT_PROMILE,
            L=LENGTH_KM * 1_000.0,
            power_system=POWER_SYSTEM,
            dx_coast=COASTING_DISTANCE_KM * 1_000.0,
        )

    def summary(self) -> str:
        """Czytelne podsumowanie parametrów (do drukowania)."""
        return (
            f"=== Parametry symulacji ===\n"
            f"  v_max          = {self.v_max * 3.6:.1f} km/h   ({self.v_max:.2f} m/s)\n"
            f"  m              = {self.m / 1000:.0f} t\n"
            f"  m_eff          = {self.m_eff / 1000:.1f} t (z masami wirującymi)\n"
            f"  P_nom          = {self.P_nom / 1e6:.1f} MW\n"
            f"  P_eff_max      = {self.P_eff_max / 1e6:.1f} MW (wg systemu zasilania)\n"
            f"  gradient       = {self.gradient:.2f} ‰\n"
            f"  L              = {self.L / 1000:.1f} km\n"
            f"  system         = {self.power_system}\n"
            f"  Δx_coast       = {self.dx_coast / 1000:.1f} km\n"
            f"  F_max          = {self.F_max / 1000:.1f} kN\n"
            f"  v_breakpoint   = {self.v_breakpoint * 3.6:.1f} km/h\n"
            f"  η_tr (effective) = {self.eta_tr_effective:.2f} (system {self.power_system})\n"
            f"  η_rec          = {self.eta_rec:.2f}\n"
            f"  η_grid         = {self.eta_grid:.2f}\n"
            f"  P_aux          = {self.P_aux / 1000:.0f} kW\n"
            f"  a_launch_max   = {self.a_launch_max:.2f} m/s²\n"
            f"  a_brake_max    = {self.a_brake_max:.2f} m/s²\n"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  🚀 SZYBKI TEST — uruchom: python parameters.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = Parameters.base()
    print(p.summary())

    # Demonstracja: kopia z innymi parametrami
    print("\n=== Kopia: 750 t, system DC ===")
    p_alt = p.with_changes(m=750_000, power_system="DC")
    print(p_alt.summary())
