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

from dataclasses import dataclass 

import numpy as np 

from engine_fast import backward_pass_njit ,forward_pass_njit 
from parameters import Parameters 
from physics import (
TrackProfile ,
)







G_CONST :float =9.81 


def _profile_to_arrays (
profile :TrackProfile ,
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """Rozbija profil [(start, end, grad), ...] na trzy tablice numpy dla Numby."""
    starts =np .array ([seg [0 ]for seg in profile ],dtype =np .float64 )
    ends =np .array ([seg [1 ]for seg in profile ],dtype =np .float64 )
    grads =np .array ([seg [2 ]for seg in profile ],dtype =np .float64 )
    return starts ,ends ,grads 







def _F_traction_vec (v :np .ndarray ,p :Parameters )->np .ndarray :
    """Wektorowa F_traction: F_max do v_1, P/v do v_2, P·v_2/v² powyżej."""
    F =np .full_like (v ,p .F_max )
    mask_power =(v >p .v_breakpoint )&(v <=p .v_field_weak )
    F [mask_power ]=p .P_eff_max /v [mask_power ]
    mask_fw =v >p .v_field_weak 
    F [mask_fw ]=p .P_eff_max *p .v_field_weak /(v [mask_fw ]*v [mask_fw ])
    return F 


def _gradient_profile_vec (x :np .ndarray ,profile :TrackProfile )->np .ndarray :
    """
    Wektorowy odczyt pochylenia [‰] dla każdego punktu x.

    Dla każdego segmentu (x_start, x_end, i) ustawia i tam gdzie x w zakresie.
    Tolerancja na końcach dla zaokrągleń float.
    """
    grad =np .zeros_like (x )
    TOL =1.0 
    for x_start ,x_end ,i_promille in profile :
        mask =(x >=x_start -TOL )&(x <=x_end +TOL )
        grad [mask ]=i_promille 
    return grad 


def _F_brake_required_vec (
v :np .ndarray ,gradient_local :np .ndarray ,p :Parameters 
)->np .ndarray :
    """
    Wektorowa F_brake_required wg sufitu przyczepności TSI 4.2.4.6.1: a_ham(v).

    F_req = m_eff·a_ham(v) - F_op(v) - F_grav  (≥ 0)
    """
    F_op =p .davis_A +p .davis_B *v +p .davis_C *v *v 
    F_grav =p .m *G_CONST *gradient_local /1000.0 

    v_kmh =v *3.6 
    mu =np .where (
    v_kmh <=250.0 ,
    p .mu_b_base ,
    p .mu_b_base -0.05 *(np .minimum (v_kmh ,350.0 )-250.0 )/100.0 ,
    )
    a_dec =mu *p .braked_frac *G_CONST 
    F_req =p .m_eff *a_dec -F_op -F_grav 
    return np .maximum (0.0 ,F_req )


@dataclass 
class SimulationProfile :
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

    x :np .ndarray 
    v :np .ndarray 
    t :np .ndarray 
    a :np .ndarray 
    phase :np .ndarray 
    F_tr :np .ndarray 
    F_brake :np .ndarray 
    F_op :np .ndarray 
    F_grav :np .ndarray 
    gradient :np .ndarray 


    x1 :float =0.0 
    x2 :float =0.0 
    x3 :float =0.0 


    reached_v_set :bool =False 

    @property 
    def T_total (self )->float :
        """Całkowity czas przejazdu [s]."""
        return float (self .t [-1 ])

    @property 
    def v_avg (self )->float :
        """Średnia prędkość [m/s]."""
        return float (self .x [-1 ]/self .t [-1 ])

    @property 
    def v_max_reached (self )->float :
        """Maksymalna osiągnięta prędkość [m/s]."""
        return float (np .max (self .v ))







def forward_pass (
p :Parameters ,profile :TrackProfile 
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    """
    Forward pass (rozpędzanie → jazda ustalona → coasting).

    Cienka nakładka na skompilowaną Numbą forward_pass_njit:
    rozpakowuje Parameters na floaty i profil na tablice.
    """
    starts ,ends ,grads =_profile_to_arrays (profile )
    return forward_pass_njit (
    p .L ,
    p .dx ,
    p .v_max ,
    p .m_eff ,
    p .F_max ,
    p .P_eff_max ,
    p .v_breakpoint ,
    p .v_field_weak ,
    p .davis_A ,
    p .davis_B ,
    p .davis_C ,
    p .dx_coast ,
    p .m ,
    starts ,
    ends ,
    grads ,
    )







def backward_pass (p :Parameters ,profile :TrackProfile )->np .ndarray :
    """
    Backward pass (hamowanie wstecz od x=L).

    Cienka nakładka na skompilowaną Numbą backward_pass_njit.
    """
    starts ,ends ,grads =_profile_to_arrays (profile )
    return backward_pass_njit (
    p .L ,
    p .dx ,
    p .m_eff ,
    p .mu_b_base ,
    p .braked_frac ,
    p .davis_A ,
    p .davis_B ,
    p .davis_C ,
    p .m ,
    starts ,
    ends ,
    grads ,
    )







def find_meeting_point (v_fwd :np .ndarray ,v_bwd :np .ndarray )->int :
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
    N =len (v_fwd )

    diff =v_fwd -v_bwd 

    candidates =np .where (diff >=0 )[0 ]
    if len (candidates )==0 :
        return N -1 
    return int (candidates [0 ])







def run_simulation (
p :Parameters ,profile :TrackProfile |None =None 
)->SimulationProfile :
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
    if profile is None :
        profile =[(0.0 ,p .L ,p .gradient )]


    x ,v_fwd ,phase_fwd =forward_pass (p ,profile )
    N =len (x )


    v_bwd =backward_pass (p ,profile )


    idx_brake_start =find_meeting_point (v_fwd ,v_bwd )
    v =np .where (np .arange (N )<idx_brake_start ,v_fwd ,v_bwd )


    phase =phase_fwd .copy ()
    phase [idx_brake_start :]=4 


    v_avg_seg =0.5 *(v [:-1 ]+v [1 :])
    v_avg_seg =np .maximum (v_avg_seg ,p .v_min_num )
    dt_seg =p .dx /v_avg_seg 
    t =np .zeros (N ,dtype =np .float64 )
    t [1 :]=np .cumsum (dt_seg )


    a =np .zeros (N ,dtype =np .float64 )
    dv =np .diff (v )
    mask_v =v [:-1 ]>p .v_min_num 
    a [:-1 ]=np .where (mask_v ,v [:-1 ]*dv /p .dx ,0.0 )
    a [-1 ]=0.0 



    F_op =p .davis_A +p .davis_B *v +p .davis_C *v *v 


    gradient_local =_gradient_profile_vec (x ,profile )
    F_grav =p .m *G_CONST *gradient_local /1000.0 
    gradient_arr =gradient_local .copy ()


    F_tr =np .zeros (N ,dtype =np .float64 )
    F_brake_total =np .zeros (N ,dtype =np .float64 )


    mask1 =phase ==1 
    F_tr [mask1 ]=_F_traction_vec (v [mask1 ],p )


    mask2 =phase ==2 
    F_tr [mask2 ]=np .maximum (0.0 ,F_op [mask2 ]+F_grav [mask2 ])




    mask4 =phase ==4 
    F_brake_total [mask4 ]=_F_brake_required_vec (v [mask4 ],gradient_local [mask4 ],p )


    x1 =float (x [np .where (phase >=2 )[0 ][0 ]])if np .any (phase >=2 )else 0.0 
    x2 =float (x [np .where (phase >=3 )[0 ][0 ]])if np .any (phase >=3 )else 0.0 
    x3 =float (x [idx_brake_start ])
    reached =bool (np .any (v_fwd >=p .v_max -0.1 ))

    return SimulationProfile (
    x =x ,
    v =v ,
    t =t ,
    a =a ,
    phase =phase ,
    F_tr =F_tr ,
    F_brake =F_brake_total ,
    F_op =F_op ,
    F_grav =F_grav ,
    gradient =gradient_arr ,
    x1 =x1 ,
    x2 =x2 ,
    x3 =x3 ,
    reached_v_set =reached ,
    )






if __name__ =="__main__":
    p =Parameters .base ()
    print (p .summary ())
    print ()

    print (">>> Uruchamiam symulację...")
    sim =run_simulation (p )

    print ()
    print ("=== Wyniki symulacji ===")
    print (f"  Czas przejazdu T   = {sim .T_total /60 :.2f} min ({sim .T_total :.0f} s)")
    print (f"  Średnia prędkość   = {sim .v_avg *3.6 :.1f} km/h")
    print (f"  v_max osiągnięte   = {sim .v_max_reached *3.6 :.1f} km/h")
    print (f"  Osiągnięto v_set?  = {sim .reached_v_set }")
    print ()
    print ("  Punkty przełączania faz:")
    print (f"    x1 (koniec rozpędzania) = {sim .x1 /1000 :.2f} km")
    print (f"    x2 (start coastingu)    = {sim .x2 /1000 :.2f} km")
    print (f"    x3 (start hamowania)    = {sim .x3 /1000 :.2f} km")
    print ()
    print ("  Pierwsze 5 i ostatnie 5 punktów profilu v(x):")
    print (f"  {'x [km]':>8} {'v [km/h]':>10} {'a [m/s²]':>10} {'faza':>6}")
    for i in list (range (5 ))+list (range (len (sim .x )-5 ,len (sim .x ))):
        print (
        f"  {sim .x [i ]/1000 :>8.3f} {sim .v [i ]*3.6 :>10.2f} {sim .a [i ]:>10.3f} {sim .phase [i ]:>6}"
        )
