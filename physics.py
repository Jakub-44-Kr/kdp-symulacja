

from __future__ import annotations 

from typing import Sequence 

from parameters import MU_B_BASE ,G ,Parameters 






TrackProfile =Sequence [tuple [float ,float ,float ]]







def F_traction (v :float ,p :Parameters )->float :
    
    if v <=p .v_breakpoint :
        return p .F_max 
    if v <=p .v_field_weak :
        return p .P_eff_max /v 
    return p .P_eff_max *p .v_field_weak /(v *v )







def F_davis (v :float ,p :Parameters )->float :
    
    return p .davis_A +p .davis_B *v +p .davis_C *v *v 







def gradient_at (x :float ,profile :TrackProfile )->float :
    
    TOL =1.0 

    for x_start ,x_end ,i_promille in profile :
        if x_start -TOL <=x <=x_end +TOL :
            return i_promille 


    if x <profile [0 ][0 ]:
        return profile [0 ][2 ]
    return profile [-1 ][2 ]


def F_gravity (x :float ,p :Parameters ,profile :TrackProfile )->float :
    
    i_promille =gradient_at (x ,profile )
    return p .m *G *i_promille /1000.0 





def mu_b (v_kmh :float ,mu_base :float =MU_B_BASE )->float :
    
    if v_kmh <=250.0 :
        return mu_base 
    v =min (v_kmh ,350.0 )
    return mu_base -0.05 *(v -250.0 )/100.0 


def a_ham (v :float ,p :Parameters )->float :
    
    return mu_b (v *3.6 ,p .mu_b_base )*p .braked_frac *G 


def F_brake_max_electric (v :float ,p :Parameters )->float :
    
    if v <p .v_brake_min :
        return 0.0 
    if v <=p .v_breakpoint :
        return p .F_max 
    if v <=p .v_field_weak :
        return p .P_eff_max /v 
    return p .P_eff_max *p .v_field_weak /(v *v )


def F_brake_required (
v :float ,
x :float ,
p :Parameters ,
profile :TrackProfile ,
target_decel :float |None =None ,
)->float :
    
    if target_decel is None :
        target_decel =a_ham (v ,p )

    F_op =F_davis (v ,p )
    F_g =F_gravity (x ,p ,profile )



    F_required =p .m_eff *target_decel -F_op -F_g 


    return max (0.0 ,F_required )


def split_brake_force (
F_required :float ,
v :float ,
p :Parameters ,
)->tuple [float ,float ]:
    
    F_el_max =F_brake_max_electric (v ,p )
    F_el =min (F_required ,F_el_max )
    F_mech =F_required -F_el 
    return F_el ,F_mech 







def F_resultant_in_phase (
phase :int ,
v :float ,
x :float ,
p :Parameters ,
profile :TrackProfile ,
)->float :
    
    if phase ==1 :

        return F_traction (v ,p )
    if phase ==2 :

        return F_davis (v ,p )+F_gravity (x ,p ,profile )
    if phase ==3 :

        return 0.0 
    if phase ==4 :

        F_req =F_brake_required (v ,x ,p ,profile )
        return -F_req 
    raise ValueError (f"Nieznana faza sterowania: {phase }")






if __name__ =="__main__":
    from parameters import Parameters 

    p =Parameters .base ()
    profile :TrackProfile =[(0.0 ,p .L ,p .gradient )]

    print ("=== Test physics.py — scenariusz bazowy ===")
    print (f"v_breakpoint = {p .v_breakpoint *3.6 :.1f} km/h")
    print ()
    print (
    f"{'v [km/h]':>10} {'F_tr [kN]':>12} {'F_davis [kN]':>14} {'F_brake_el_max [kN]':>22}"
    )
    print ("-"*60 )
    for v_kmh in [0 ,30 ,50 ,71 ,100 ,150 ,200 ,250 ,320 ,400 ]:
        v_ms =v_kmh /3.6 
        Ft =F_traction (v_ms ,p )/1000.0 
        Fd =F_davis (v_ms ,p )/1000.0 
        Fb =F_brake_max_electric (v_ms ,p )/1000.0 
        print (f"{v_kmh :>10} {Ft :>12.1f} {Fd :>14.2f} {Fb :>22.1f}")

    print ()
    print ("=== Test pochyleń ===")
    profile_alt :TrackProfile =[
    (0.0 ,50_000.0 ,0.0 ),
    (50_000.0 ,100_000.0 ,5.0 ),
    (100_000.0 ,p .L ,-3.0 ),
    ]
    for x_km in [0 ,25 ,50 ,75 ,100 ,125 ,180 ]:
        x_m =x_km *1000.0 
        i =gradient_at (x_m ,profile_alt )
        Fg =F_gravity (x_m ,p ,profile_alt )/1000.0 
        print (f"x = {x_km :>4} km   i = {i :+.1f}‰   F_g = {Fg :+8.2f} kN")
