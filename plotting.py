"""
plotting.py — Wykresy wyników symulacji pociągu KDP.

Generuje zestaw wykresów do analizy i prezentacji w pracy magisterskiej.
Wszystkie wykresy zapisywane są w folderze outputs/ jako PNG (300 dpi).

Każdy wykres ma osobną funkcję - łatwo dodawać nowe lub wyłączać.

Autor: Jakub Król, PW WE, 2026
"""

from __future__ import annotations 

from pathlib import Path 

import matplotlib .pyplot as plt 
import numpy as np 

from energy import EnergyResults 
from parameters import OUTPUT_DIR ,Parameters 
from simulation import SimulationProfile 





plt .rcParams .update (
{
"font.family":"serif",
"font.size":11 ,
"axes.titlesize":12 ,
"axes.labelsize":11 ,
"xtick.labelsize":10 ,
"ytick.labelsize":10 ,
"legend.fontsize":10 ,
"figure.dpi":100 ,
"savefig.dpi":300 ,
"savefig.bbox":"tight",
"axes.grid":True ,
"grid.alpha":0.3 ,
"grid.linestyle":"--",
"axes.spines.top":False ,
"axes.spines.right":False ,
}
)


PHASE_COLORS ={
1 :"#1f77b4",
2 :"#2ca02c",
3 :"#ff7f0e",
4 :"#d62728",
}
PHASE_NAMES ={
1 :"Rozpędzanie",
2 :"Jazda ustalona",
3 :"Wybieg (coasting)",
4 :"Hamowanie",
}







def _trim_artifacts (arr :np .ndarray ,n_trim :int =3 )->np .ndarray :
    """Zwraca tablicę z ostatnich n_trim punktów zamienionych na NaN.
    Używane do ukrycia artefaktów numerycznych w wykresach a(x)."""
    result =arr .copy ().astype (float )
    if len (result )>n_trim :
        result [-n_trim :]=np .nan 
    return result 


def _add_phase_regions (ax ,sim :SimulationProfile ,alpha :float =0.20 )->None :
    """Koloruje tło wg faz sterowania.

    alpha podniesione z 0,08 → 0,20 (uwaga promotora: kolory faz słabo widoczne).
    Czarna krzywa prędkości (zorder=3) pozostaje wyraźna nad tłem (zorder=0).
    """
    phases =sim .phase 
    x_km =sim .x /1000.0 
    current_phase =phases [0 ]
    start_idx =0 
    for i in range (1 ,len (phases )):
        if phases [i ]!=current_phase :
            ax .axvspan (
            x_km [start_idx ],
            x_km [i ],
            color =PHASE_COLORS [int (current_phase )],
            alpha =alpha ,
            zorder =0 ,
            )
            current_phase =phases [i ]
            start_idx =i 

    ax .axvspan (
    x_km [start_idx ],
    x_km [-1 ],
    color =PHASE_COLORS [int (current_phase )],
    alpha =alpha ,
    zorder =0 ,
    )


def _legend_phases (ax )->None :
    """Dodaje legendę dla faz (kolorowe prostokąty)."""
    from matplotlib .patches import Patch 

    handles =[
    Patch (facecolor =PHASE_COLORS [p ],alpha =0.45 ,label =PHASE_NAMES [p ])
    for p in (1 ,2 ,3 ,4 )
    ]
    ax .legend (handles =handles ,loc ="best",framealpha =0.9 )







def plot_velocity_profile (
sim :SimulationProfile ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Profil prędkości v(x) z zaznaczonymi fazami sterowania."""
    fig ,ax =plt .subplots (figsize =(10 ,5 ))

    _add_phase_regions (ax ,sim )

    x_km =sim .x /1000.0 
    v_kmh =sim .v *3.6 
    ax .plot (x_km ,v_kmh ,color ="#000000",linewidth =1.5 ,zorder =3 )


    ax .axhline (
    p .v_max *3.6 ,
    color ="gray",
    linestyle =":",
    linewidth =1 ,
    label =f"$v_{{max}}$ = {p .v_max *3.6 :.0f} km/h",
    )


    for x_switch ,label in [(sim .x1 ,"$x_1$"),(sim .x2 ,"$x_2$"),(sim .x3 ,"$x_3$")]:
        ax .axvline (
        x_switch /1000.0 ,color ="gray",linestyle ="--",linewidth =0.8 ,alpha =0.6 
        )
        ax .text (
        x_switch /1000.0 ,p .v_max *3.6 *1.02 ,label ,ha ="center",fontsize =10 
        )

    ax .set_xlabel ("Pozycja $x$ [km]")
    ax .set_ylabel ("Prędkość $v$ [km/h]")
    ax .set_title (
    f"Profil prędkości — {p .v_max *3.6 :.0f} km/h, "
    f"{p .m /1000 :.0f} t, {p .P_nom /1e6 :.0f} MW, "
    f"L = {p .L /1000 :.0f} km, system {p .power_system }"
    )
    ax .set_xlim (0 ,p .L /1000.0 )
    ax .set_ylim (0 ,p .v_max *3.6 *1.1 )
    _legend_phases (ax )

    path =save_dir /"01_velocity_profile.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_forces (
sim :SimulationProfile ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Wykres sił: trakcyjna, opory, grawitacja, hamowanie."""
    fig ,ax =plt .subplots (figsize =(10 ,5 ))

    _add_phase_regions (ax ,sim )

    x_km =sim .x /1000.0 
    ax .plot (
    x_km ,
    sim .F_tr /1000.0 ,
    label ="$F_{tr}$ (trakcja)",
    color ="#1f77b4",
    linewidth =1.5 ,
    )
    ax .plot (
    x_km ,
    -sim .F_brake /1000.0 ,
    label ="$-F_{ham}$ (hamowanie)",
    color ="#d62728",
    linewidth =1.5 ,
    )
    ax .plot (
    x_km ,
    sim .F_op /1000.0 ,
    label ="$F_{op}$ (opory Davisa)",
    color ="#7f7f7f",
    linewidth =1.2 ,
    linestyle ="--",
    )
    ax .plot (
    x_km ,
    sim .F_grav /1000.0 ,
    label ="$F_{g}$ (grawitacja)",
    color ="#9467bd",
    linewidth =1.2 ,
    linestyle =":",
    )

    ax .axhline (0 ,color ="black",linewidth =0.5 )

    ax .set_xlabel ("Pozycja $x$ [km]")
    ax .set_ylabel ("Siła [kN]")
    ax .set_title ("Siły działające na pociąg wzdłuż trasy")
    ax .set_xlim (0 ,p .L /1000.0 )
    ax .legend (loc ="best",framealpha =0.9 )

    path =save_dir /"02_forces.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_power_current (
sim :SimulationProfile ,
energy :EnergyResults ,
p :Parameters ,
save_dir :Path =OUTPUT_DIR ,
)->Path :
    """Moc (pobierana, zwracana, netto) i prąd na pantografie."""
    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(10 ,8 ),sharex =True )

    _add_phase_regions (ax1 ,sim )
    _add_phase_regions (ax2 ,sim )

    x_km =sim .x /1000.0 


    ax1 .plot (
    x_km ,
    energy .P_pant_draw /1e6 ,
    label ="$P_{pant}$ pobierana",
    color ="#1f77b4",
    linewidth =1.5 ,
    )
    ax1 .plot (
    x_km ,
    -energy .P_pant_rec /1e6 ,
    label ="$P_{pant}$ zwracana",
    color ="#2ca02c",
    linewidth =1.5 ,
    )
    ax1 .plot (
    x_km ,
    energy .P_pant_net /1e6 ,
    label ="$P_{pant}$ netto",
    color ="black",
    linewidth =1.0 ,
    linestyle ="--",
    alpha =0.7 ,
    )


    ax1 .axhline (
    p .P_eff_max /1e6 ,
    color ="red",
    linestyle =":",
    linewidth =1 ,
    label =f"$P_{{eff,max}}$ = {p .P_eff_max /1e6 :.1f} MW",
    )

    ax1 .axhline (0 ,color ="black",linewidth =0.5 )
    ax1 .set_ylabel ("Moc na pantografie [MW]")
    ax1 .set_title (f"Moc na pantografie i prąd — system {p .power_system }")
    ax1 .legend (loc ="best",framealpha =0.9 )


    ax2 .plot (x_km ,energy .I_pant ,color ="#d62728",linewidth =1.2 )
    ax2 .axhline (
    p .I_grid_limit ,
    color ="red",
    linestyle =":",
    linewidth =1 ,
    label =f"limit TSI ENE = {p .I_grid_limit :.0f} A",
    )
    ax2 .set_xlabel ("Pozycja $x$ [km]")
    ax2 .set_ylabel (f"|$I_{{pant}}$| [A] @ U = {p .U_grid /1000 :.0f} kV")
    ax2 .legend (loc ="best",framealpha =0.9 )
    ax2 .set_xlim (0 ,p .L /1000.0 )

    path =save_dir /"03_power_current.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_velocity_time (
sim :SimulationProfile ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Profil prędkości w funkcji czasu v(t)."""
    fig ,ax =plt .subplots (figsize =(10 ,5 ))

    t_min =sim .t /60.0 
    v_kmh =sim .v *3.6 
    ax .plot (t_min ,v_kmh ,color ="#000000",linewidth =1.5 )
    ax .axhline (
    p .v_max *3.6 ,
    color ="gray",
    linestyle =":",
    linewidth =1 ,
    label =f"$v_{{max}}$ = {p .v_max *3.6 :.0f} km/h",
    )

    ax .set_xlabel ("Czas $t$ [min]")
    ax .set_ylabel ("Prędkość $v$ [km/h]")
    ax .set_title (f"Profil prędkości w czasie — T = {sim .T_total /60 :.2f} min")
    ax .set_xlim (0 ,sim .T_total /60 )
    ax .set_ylim (0 ,p .v_max *3.6 *1.1 )
    ax .legend (loc ="best",framealpha =0.9 )

    path =save_dir /"04_velocity_time.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_acceleration (
sim :SimulationProfile ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Profil przyspieszenia a(x) z zaznaczonymi limitami komfortu."""
    fig ,ax =plt .subplots (figsize =(10 ,5 ))

    _add_phase_regions (ax ,sim )

    x_km =sim .x /1000.0 

    a_clean =_trim_artifacts (sim .a ,n_trim =3 )
    ax .plot (x_km ,a_clean ,color ="#000000",linewidth =1.2 )


    ax .axhline (
    p .a_launch_max ,
    color ="green",
    linestyle =":",
    linewidth =1 ,
    label =f"$a_{{rozr,max}}$ = {p .a_launch_max } m/s²",
    )
    ax .axhline (
    -p .a_brake_max ,
    color ="red",
    linestyle =":",
    linewidth =1 ,
    label =f"$a_{{ham,max}}$ = -{p .a_brake_max } m/s²",
    )
    ax .axhline (0 ,color ="black",linewidth =0.5 )

    ax .set_xlabel ("Pozycja $x$ [km]")
    ax .set_ylabel ("Przyspieszenie $a$ [m/s²]")
    ax .set_title ("Profil przyspieszenia wzdłuż trasy")
    ax .set_xlim (0 ,p .L /1000.0 )
    ax .legend (loc ="best",framealpha =0.9 )

    path =save_dir /"05_acceleration.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_energy_balance (
energy :EnergyResults ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Słupkowy wykres bilansu energii."""
    from energy import J_TO_KWH 

    fig ,ax =plt .subplots (figsize =(10 ,6 ))


    labels =[
    "$E_{trakcja,pant}$",
    "$E_{aux}$",
    "$E_{pant,pobrana}$",
    "$E_{rec,pant}$",
    "$E_{pant,NETTO}$",
    ]
    values =[
    energy .E_trakcja_pant *J_TO_KWH ,
    energy .E_aux *J_TO_KWH ,
    energy .E_pant_pobrana *J_TO_KWH ,
    -energy .E_rec_pant *J_TO_KWH ,
    energy .E_pant_netto *J_TO_KWH ,
    ]
    colors =["#1f77b4","#ff7f0e","#7f7f7f","#2ca02c","#d62728"]

    bars =ax .bar (labels ,values ,color =colors ,edgecolor ="black",alpha =0.85 )


    for bar ,val in zip (bars ,values ):
        height =bar .get_height ()
        y_pos =(
        height +(max (values )-min (values ))*0.02 
        if height >=0 
        else height -(max (values )-min (values ))*0.05 
        )
        ax .text (
        bar .get_x ()+bar .get_width ()/2 ,
        y_pos ,
        f"{val :+.1f}",
        ha ="center",
        fontsize =10 ,
        fontweight ="bold"if "NETTO"in bar .get_label ()else "normal",
        )

    ax .axhline (0 ,color ="black",linewidth =0.8 )
    ax .set_ylabel ("Energia [kWh]")
    ax .set_title (
    f"Bilans energetyczny przejazdu — system {p .power_system }\n"
    f"$E_{{per\\_km}}$ = {energy .E_per_km :.1f} kWh/km   "
    f"$\\approx$ {energy .E_per_seat_km :.1f} Wh/(pas$\\cdot$km)"
    )

    path =save_dir /"06_energy_balance.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_track_profile (
sim :SimulationProfile ,p :Parameters ,save_dir :Path =OUTPUT_DIR 
)->Path :
    """Profil pochyleń trasy oraz odpowiadająca składowa grawitacyjna."""
    fig ,(ax1 ,ax2 )=plt .subplots (2 ,1 ,figsize =(10 ,6 ),sharex =True )

    x_km =sim .x /1000.0 
    ax1 .plot (x_km ,sim .gradient ,color ="#9467bd",linewidth =1.5 )
    ax1 .axhline (0 ,color ="black",linewidth =0.5 )
    ax1 .fill_between (
    x_km ,
    0 ,
    sim .gradient ,
    where =(sim .gradient >=0 ),
    color ="#9467bd",
    alpha =0.3 ,
    label ="wzniesienie",
    )
    ax1 .fill_between (
    x_km ,
    0 ,
    sim .gradient ,
    where =(sim .gradient <0 ),
    color ="#bcbd22",
    alpha =0.3 ,
    label ="spadek",
    )
    ax1 .set_ylabel ("Pochylenie $i$ [‰]")
    ax1 .set_title ("Profil trasy")
    ax1 .legend (loc ="best")

    ax2 .plot (x_km ,sim .F_grav /1000.0 ,color ="#9467bd",linewidth =1.5 )
    ax2 .axhline (0 ,color ="black",linewidth =0.5 )
    ax2 .set_xlabel ("Pozycja $x$ [km]")
    ax2 .set_ylabel ("$F_g$ [kN]")
    ax2 .set_xlim (0 ,p .L /1000.0 )

    path =save_dir /"07_track_profile.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_traction_characteristic (p :Parameters ,save_dir :Path =OUTPUT_DIR )->Path :
    """Charakterystyka trakcyjna F_tr(v) i F_ham_max,el(v) - rys. 3 z pracy."""
    from physics import F_brake_max_electric ,F_traction 

    fig ,ax =plt .subplots (figsize =(10 ,5 ))

    v_range_ms =np .linspace (0.1 ,p .v_max *1.2 ,500 )
    F_tr_arr =np .array ([F_traction (v ,p )for v in v_range_ms ])/1000.0 
    F_ham_arr =np .array ([F_brake_max_electric (v ,p )for v in v_range_ms ])/1000.0 

    ax .plot (
    v_range_ms *3.6 ,F_tr_arr ,label ="$F_{tr}(v)$",color ="#1f77b4",linewidth =2 
    )
    ax .plot (
    v_range_ms *3.6 ,
    -F_ham_arr ,
    label ="$-F_{ham,el,max}(v)$",
    color ="#d62728",
    linewidth =2 ,
    )

    ax .axhline (0 ,color ="black",linewidth =0.5 )
    ax .axvline (
    p .v_breakpoint *3.6 ,
    color ="gray",
    linestyle ="--",
    linewidth =0.8 ,
    label =f"$v_b$ = {p .v_breakpoint *3.6 :.0f} km/h",
    )
    ax .axvline (
    p .v_max *3.6 ,
    color ="gray",
    linestyle =":",
    linewidth =0.8 ,
    label =f"$v_{{max}}$ = {p .v_max *3.6 :.0f} km/h",
    )

    ax .set_xlabel ("Prędkość $v$ [km/h]")
    ax .set_ylabel ("Siła [kN]")
    ax .set_title (
    f"Charakterystyka trakcyjna i hamowania elektrycznego — "
    f"{p .m /1000 :.0f} t, {p .P_nom /1e6 :.0f} MW"
    )
    ax .legend (loc ="best",framealpha =0.9 )

    path =save_dir /"08_traction_characteristic.png"
    fig .savefig (path )
    plt .close (fig )
    return path 







def plot_all (
sim :SimulationProfile ,
energy :EnergyResults ,
p :Parameters ,
save_dir :Path =OUTPUT_DIR ,
)->list [Path ]:
    """Wygeneruj wszystkie standardowe wykresy."""
    save_dir .mkdir (parents =True ,exist_ok =True )
    paths =[
    plot_velocity_profile (sim ,p ,save_dir ),
    plot_forces (sim ,p ,save_dir ),
    plot_power_current (sim ,energy ,p ,save_dir ),
    plot_velocity_time (sim ,p ,save_dir ),
    plot_acceleration (sim ,p ,save_dir ),
    plot_energy_balance (energy ,p ,save_dir ),
    plot_track_profile (sim ,p ,save_dir ),
    plot_traction_characteristic (p ,save_dir ),
    ]
    return paths 






if __name__ =="__main__":
    from energy import compute_energy 
    from simulation import run_simulation 

    p =Parameters .base ()
    print (">>> Symulacja...")
    sim =run_simulation (p )
    print (">>> Bilans energii...")
    energy =compute_energy (sim ,p )
    print (">>> Generuję wykresy...")
    paths =plot_all (sim ,energy ,p )
    print ()
    print (f"Zapisano {len (paths )} wykresów w {OUTPUT_DIR }:")
    for path in paths :
        print (f"  - {path .name }")
