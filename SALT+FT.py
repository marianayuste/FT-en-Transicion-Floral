# Código para ingresar red actualizada a un modelo 3D
# Hecho por Mariana Yuste
# 2025

import numpy as np
from copy import deepcopy
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------- #
#                                  PARÁMETROS                                  #
# ---------------------------------------------------------------------------- #
# de la red Azpeitia et al., (2021):
n=2
deg = 0.1
kSL = 6.
kTL = 0.5
kAL = 4.0
kLL = 100
kauxL = 1
kLA = 1.0
kTA = 1.0
kFA = 50.0
kSA = 15.0
kAS = 4.
kFS = 10.
kSS = 20.
kST = 1.
kLT = 100.0
kSAT = 1.
kRT = 0.1
kDimAP1SAS=0.1
kDisAP1SAS=0.01
bauxL = np.sqrt(0.2)

# Definidos manualmente para la inclusión de FT:
DimFF = 37
DisFF = 37
DimTF = 55
DisTF = 0.0055
kFAc = 1
kTAc = 1
kFLc = 0.1
kTLc = 0.01


def dX_dt_comp(t,y):
    '''
    Sistema de ecuaciones de SALT+FT
    Con complejos FT-FD(CFF) y TFL1-FD(CTF)
    Con competencia inhibitoria entre ellos
    '''

    L,A,S,SA,T,aux,R,FD,CTF,CTF,FT=y

    lfy_ODE  = (CTF/(kFLc*(1+(CTF/kTLc))+CTF))*(((aux+bauxL)/(aux+bauxL+kauxL))*(kLL**n/(L**n+kLL**n))*(S**n/(S**n+kSL**n))+(A**n/(A**n+kAL**n)))-deg*L
    ap1_ODE  = (CTF/(kFAc*(1+(CTF/kTAc))+CTF))*((L**n/(L**n+kLA**n))+(S**n/(S**n+kSA**n)))+((-kDimAP1SAS*A*S)+(kDisAP1SAS*SA))-deg*A
    sas_ODE  = (kAS**n/(A**n+kAS**n))*((CTF/(CTF+kFS))+(S**n/(S**n+kSS**n)))+((-kDimAP1SAS*A*S)+(kDisAP1SAS*SA))-deg*S
    tfl1_ODE  = ((S**n/(S**n+kST**n))*((kRT/(kRT+R))*(kSAT**n/(SA**n+kSAT**n))))+(L**n/(L**n+kLT**n))+((-DimTF*T*FD)+(DisTF*CTF))-deg*T
    fd_ODE = (-DimTF*T*FD)+(DisTF*CTF)+(DisFF*CTF)-(DimFF*FT*FD)+(deg*CTF)
    Ctf_ODE = (DimTF*T*FD)-(DisTF*CTF)-(deg*CTF)
    Cff_ODE = (DimFF*FT*FD)-(DisFF*CTF)
    return np.array([lfy_ODE,
            ap1_ODE,
            sas_ODE,
            ((kDimAP1SAS*A*S)-(kDisAP1SAS*SA))-deg*SA, # SA
            tfl1_ODE,
            -deg*aux,
            -deg*R,
            fd_ODE,
            Ctf_ODE,
            Cff_ODE,
            (DisFF*CTF)-(DimFF*FT*FD) # FT
            ])

def sim_comp(X0):
    '''Devuelve puntos finales del sistema de ecuaciones de SALT+FT'''

    t = [0,200]
    X = solve_ivp(dX_dt_comp,t,X0,method='LSODA')
    final_points = np.zeros(11)
    gene_counter = 0
    for i in X.y:
        final_points[gene_counter] = i[-1]
        gene_counter += 1
    return final_points

