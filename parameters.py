"""
parameters.py — Parametry modelu symulacyjnego pociągu KDP.

Plik centralny zawierający wszystkie parametry fizyczne, operacyjne
i numeryczne modelu. Pozostałe moduły importują parametry stąd.

Konwencja jednostek: WSZYSTKO w SI (kg, m, s, N, W, V, A).
Pochylenie wewnętrznie przechowywane jest w promilach [‰] - konwersja
na sin(theta) wykonywana jest w physics.py wg wzoru (24).

Odwołania do wzorów odnoszą się do pracy magisterskiej.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

from dataclasses import dataclass ,field ,replace 
from pathlib import Path 







V_MAX_KMH :float =250.0 
MASS_TON :float =600.0 
POWER_MW :float =(
12.0 
)
GRADIENT_PROMILE :float =0.0 
LENGTH_KM :float =100.0 



POWER_SYSTEM :str ="AC"


COASTING_DISTANCE_KM :float =(
5.0 
)







G :float =9.81 
ROTATING_MASS_FACTOR :float =0.08 































MASS_BASE_KG :float =600_000.0 


DAVIS_A_BASE :float =2200.0 
DAVIS_B_BASE :float =130.0 
DAVIS_A_PER_KG :float =DAVIS_A_BASE /MASS_BASE_KG 
DAVIS_B_PER_KG :float =DAVIS_B_BASE /MASS_BASE_KG 


DAVIS_C0 :float =2.85 
DAVIS_C_PER_KG :float =0.0060 /1000.0 



def davis_A_of_mass (m_kg :float )->float :
    """Współczynnik A Davisa [N] jako funkcja masy. Proporcjonalny: A(m) = A₀·m (przez 0)."""
    return DAVIS_A_PER_KG *m_kg 


def davis_B_of_mass (m_kg :float )->float :
    """Współczynnik B Davisa [N·s/m] jako funkcja masy. Proporcjonalny: B(m) = B₀·m (przez 0)."""
    return DAVIS_B_PER_KG *m_kg 


def davis_C_of_mass (m_kg :float )->float :
    """Współczynnik C Davisa [N·s²/m²] jako funkcja masy.

    Afiniczny: C(m) = C₀ + c·m. Niezerowy wyraz wolny C₀ (opór nosa, ogona
    i pantografu) jest niezależny od masy — to JEDYNA różnica względem A i B,
    które przechodzą przez zero.
    """
    return DAVIS_C0 +DAVIS_C_PER_KG *m_kg 



DAVIS_A :float =davis_A_of_mass (MASS_BASE_KG )
DAVIS_B :float =davis_B_of_mass (MASS_BASE_KG )
DAVIS_C :float =davis_C_of_mass (MASS_BASE_KG )






ETA_TR_AC :float =0.86 
ETA_TR_DC :float =0.82 
ETA_TR :float =0.88 
ETA_REC :float =(
0.85 
)
ETA_GRID_AC :float =1.0 
ETA_GRID_DC :float =0.0 

ETA_REC_EFF_AC :float =0.80 
ETA_REC_EFF_DC :float =0.15 


P_AUX_KW :float =450.0 


P_MAX_AC_MW :float =12.0 
P_MAX_DC_MW :float =9.0 


U_GRID_AC :float =25_000.0 
U_GRID_DC :float =3_000.0 


I_MAX_AC :float =600.0 
I_MAX_DC :float =4_000.0 


A_BRAKE_MAX :float =(
1.2 
)
V_BRAKE_MIN_KMH :float =(
10.0 
)





MU_B_BASE :float =0.15 
BRAKED_MASS_FRACTION :float =1.0 


A_LAUNCH_MAX :float =0.7 

MU_ADHESION :float =0.30 
ADHESION_MASS_FACTOR :float =(
0.50 
)








V2_FIELD_WEAKENING_KMH :float =300.0 






DX_STEP :float =1.0 


V_MIN_NUMERICAL :float =0.1 



PROJECT_ROOT :Path =Path (__file__ ).parent 
OUTPUT_DIR :Path =PROJECT_ROOT /"outputs"
OUTPUT_DIR .mkdir (exist_ok =True )









FIG_FORMAT :str ="png"


def save_figure (fig ,save_dir :Path ,name :str ,fmt :str |None =None ):
    """
    Zapis rysunku w formacie wg globalnego FIG_FORMAT (lub nadpisanym `fmt`).

    `name` bez rozszerzenia. Linie z wielką liczbą punktów (profile ~180k)
    warto wcześniej oznaczyć .set_rasterized(True) — wtedy w PDF/SVG sama
    linia jest rastrem 300 dpi, a osie/opisy pozostają wektorem.
    Zwraca Path zapisanego pliku.
    """
    ext =(fmt or FIG_FORMAT ).lower ().lstrip (".")
    save_dir .mkdir (parents =True ,exist_ok =True )
    path =save_dir /f"{name }.{ext }"
    fig .savefig (path )
    return path 







@dataclass (frozen =True )
class Parameters :
    """
    Komplet parametrów symulacji w jednostkach SI.

    Klasa frozen — niemodyfikowalna po utworzeniu. Aby zmienić parametr
    w analizie wrażliwości, użyj metody .with_changes() która zwraca KOPIĘ.

    Example:
        >>> p = Parameters.base()
        >>> p_heavy = p.with_changes(m=750_000)  # 750 t zamiast 600 t
    """


    v_max :float 
    m :float 
    P_nom :float 
    gradient :float 
    L :float 


    power_system :str 


    dx_coast :float 






    eta_tr :float =ETA_TR 
    eta_rec :float =ETA_REC 
    regen :bool =True 
    P_aux :float =field (default =P_AUX_KW *1e3 )
    a_brake_max :float =A_BRAKE_MAX 
    mu_b_base :float =MU_B_BASE 
    braked_frac :float =BRAKED_MASS_FRACTION 
    v_brake_min :float =field (default =V_BRAKE_MIN_KMH /3.6 )
    a_launch_max :float =A_LAUNCH_MAX 
    v_field_weak :float =field (
    default =V2_FIELD_WEAKENING_KMH /3.6 
    )
    rot_mass_factor :float =ROTATING_MASS_FACTOR 


    dx :float =DX_STEP 
    v_min_num :float =V_MIN_NUMERICAL 



    @property 
    def m_eff (self )->float :
        """Masa efektywna z uwzględnieniem mas wirujących [kg]."""
        return self .m *(1.0 +self .rot_mass_factor )






    @property 
    def davis_A (self )->float :
        """Współczynnik A Davisa [N] dla masy scenariusza (A ∝ m, przez 0)."""
        return davis_A_of_mass (self .m )

    @property 
    def davis_B (self )->float :
        """Współczynnik B Davisa [N·s/m] dla masy scenariusza (B ∝ m, przez 0)."""
        return davis_B_of_mass (self .m )

    @property 
    def davis_C (self )->float :
        """Współczynnik C Davisa [N·s²/m²] dla masy scenariusza (C = C₀ + c·m, afiniczny)."""
        return davis_C_of_mass (self .m )

    @property 
    def P_eff_max (self )->float :
        """
        Efektywna max moc na pantografie wg wzoru (33) z pracy.
        Dla DC ograniczona do 6 MW, dla AC do 12 MW.
        """

        P_cap =P_MAX_DC_MW *1e6 if self .power_system =="DC"else P_MAX_AC_MW *1e6 
        return (min (self .P_nom ,P_cap )-self .P_aux )*self .eta_tr_effective 

    @property 
    def eta_grid (self )->float :
        """Receptywność sieci na energię rekuperowaną."""
        return ETA_GRID_DC if self .power_system =="DC"else ETA_GRID_AC 

    @property 
    def eta_rec_eff (self )->float :
        """(A) Efektywna sprawność rekuperacji, jedna liczba/system. regen=False => 0."""
        if not self .regen :
            return 0.0 
        return ETA_REC_EFF_DC if self .power_system =="DC"else ETA_REC_EFF_AC 

    @property 
    def U_grid (self )->float :
        """Napięcie referencyjne sieci [V] (do liczenia prądu pantografu)."""
        return U_GRID_DC if self .power_system =="DC"else U_GRID_AC 

    @property 
    def I_grid_limit (self )->float :
        """Limit prądu pantografu wg TSI ENE [A] (do walidacji)."""
        return I_MAX_DC if self .power_system =="DC"else I_MAX_AC 

    @property 
    def eta_tr_effective (self )->float :
        """
        Efektywna sprawność toru przekazywania mocy zależnie od systemu zasilania.

        DC ma niższą sprawność z powodu znacznie wyższych prądów (4 kA vs 400 A)
        i wynikających z tego strat I²R w sieci trakcyjnej (Steimel 2008).
        """
        return ETA_TR_DC if self .power_system =="DC"else ETA_TR_AC 

    @property 
    def F_max (self )->float :
        """
        Maksymalna siła trakcyjna [N] - mniejsza z dwóch:
          - ograniczenie adhezji: μ · m_ad · g
          - ograniczenie komfortu: (m + m_rot) · a_launch_max

        Patrz rozdz. 4.2 oraz 5.3 pracy.
        """
        F_adhesion =MU_ADHESION *(ADHESION_MASS_FACTOR *self .m )*G 
        F_comfort =self .m_eff *self .a_launch_max 
        return min (F_adhesion ,F_comfort )

    @property 
    def v_breakpoint (self )->float :
        """
        Prędkość łamania charakterystyki F(v): od F=const do F=P/v [m/s].
        v_b = P_eff_max / F_max
        """
        return self .P_eff_max /self .F_max 

    def with_changes (self ,**kwargs )->"Parameters":
        """
        Zwraca KOPIĘ z nadpisanymi wybranymi parametrami.
        Używane do analizy wrażliwości.

        Example:
            >>> p_base = Parameters.base()
            >>> p_dc = p_base.with_changes(power_system="DC")
            >>> p_fast = p_base.with_changes(v_max=400/3.6)
        """
        return replace (self ,**kwargs )

    @classmethod 
    def base (cls )->"Parameters":
        """Parametry scenariusza bazowego (Tabela 4)."""
        return cls (
        v_max =V_MAX_KMH /3.6 ,
        m =MASS_TON *1_000.0 ,
        P_nom =POWER_MW *1e6 ,
        gradient =GRADIENT_PROMILE ,
        L =LENGTH_KM *1_000.0 ,
        power_system =POWER_SYSTEM ,
        dx_coast =COASTING_DISTANCE_KM *1_000.0 ,
        )

    def summary (self )->str :
        """Czytelne podsumowanie parametrów (do drukowania)."""
        return (
        f"=== Parametry symulacji ===\n"
        f"  v_max          = {self .v_max *3.6 :.1f} km/h   ({self .v_max :.2f} m/s)\n"
        f"  m              = {self .m /1000 :.0f} t\n"
        f"  m_eff          = {self .m_eff /1000 :.1f} t (z masami wirującymi)\n"
        f"  P_nom          = {self .P_nom /1e6 :.1f} MW\n"
        f"  P_eff_max      = {self .P_eff_max /1e6 :.1f} MW (wg systemu zasilania)\n"
        f"  gradient       = {self .gradient :.2f} ‰\n"
        f"  L              = {self .L /1000 :.1f} km\n"
        f"  system         = {self .power_system }\n"
        f"  Δx_coast       = {self .dx_coast /1000 :.1f} km\n"
        f"  F_max          = {self .F_max /1000 :.1f} kN\n"
        f"  v_breakpoint   = {self .v_breakpoint *3.6 :.1f} km/h (v_1)\n"
        f"  v_field_weak   = {self .v_field_weak *3.6 :.1f} km/h (v_2, osłabianie pola)\n"
        f"  Davis A/B/C    = {self .davis_A :.1f} N / {self .davis_B :.2f} N·s/m / {self .davis_C :.3f} N·s²/m² (z masy {self .m /1000 :.0f} t)\n"
        f"  η_tr (effective) = {self .eta_tr_effective :.2f} (system {self .power_system })\n"
        f"  η_rec          = {self .eta_rec :.2f}\n"
        f"  η_grid         = {self .eta_grid :.2f}\n"
        f"  P_aux          = {self .P_aux /1000 :.0f} kW\n"
        f"  a_launch_max   = {self .a_launch_max :.2f} m/s²\n"
        f"  a_brake_max    = {self .a_brake_max :.2f} m/s²\n"
        )






if __name__ =="__main__":
    p =Parameters .base ()
    print (p .summary ())


    print ("\n=== Kopia: 750 t, system AC ===")
    p_alt =p .with_changes (m =750_000 ,power_system ="AC")
    print (p_alt .summary ())
