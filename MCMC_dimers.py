# ---------------------------------------------------------------------------- #
#               Análisis MCMC para parámetros ya definidos a mano              #
# ---------------------------------------------------------------------------- #
# Parámetros a buscar son 
#   los de los dímeros DFF y DTF 
#   y los de la competencia inhibitoria:
#   DimFF,DisFF,DimTF,DisTF,kFAc,kTAc,kFLc,kTLc

# Los resultados generados en el hub de matmor se guardarán en resultados_MCMC_dimers

import emcee
import numpy as np
from scipy.integrate import solve_ivp

# ---- Constantes para tener en cuenta (y subir en analizar_resultados.py) --- #
intento = 1
    # En el matmor se correrán 10 ejecuciones (ver Coliflor Notas/MCMC revancha/5ta prueba)
n_walkers = 100 #total de caminatas aleatorias que se harán
n_steps = 2000 #total de pasos que dará cada caminata
ndim = 8 # total de parámetros a buscar


# Definir el alpha (Porcentaje de variación para cada parámetro (definirá la desviación estándar))
alpha = 0.5

# -------------------------- Parámetros ya definidos ------------------------- #
# PARÁMETROS ORIGINALES:
n=2
deg = 0.1
kSL = 6.#5.#this parameter changes the threshold for the flower transition in the mutant tfl1
kTL = 0.5#0.8#0.35#this parameter changes the threshold for the flower transition in the wt plant
kAL = 4.0
kLL = 100
kauxL = 1
kLA = 1.0
kTA = 1.0
kFA = 50.0
kSA = 15.0
kAS = 4.#5.
kFS = 10.#15.
kSS = 20.#25.
kST = 1.#1.#2.
kLT = 100.0
kSAT = 1.#15.
kRT = 0.1
kDimAP1SAS=0.1
kDisAP1SAS=0.01
bauxL = np.sqrt(0.2) #np.sqrt(0.1) en codigo publicado #Independent LFY induction by Auxin


# ------------------------ Otras constantes del modelo ----------------------- #
# condiciones de cada dimensión de los que me interesan saber los puntos de equilibrio
Fs = range(11)
initial_auxin = [0,10]

# Valores iniciales constantes:
lfy = 0.2
ap1 = 0
sax = 0
dsa = 0 #antes sap
tfl1 = 0
fd = 10
dtf = 0
dff = 0

#tiempo
end = 150
t = [0,end]

# Para obtener las diferencias para el likelihood:
def delta(big,small):
    if big > small:
        return 0
    else:
        return small-big


# -------------------------- Log likelihood function ------------------------- #
def log_likelihood(theta):
    DimFF,DisFF,DimTF,DisTF,kFAc,kTAc,kFLc,kTLc = theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7]
    params = [DimFF,DisFF,DimTF,DisTF,kFAc,kTAc,kFLc,kTLc]

    # Define the model (EDOs con competencia inhibitoria)
    def dX_dt_comp(t,y,params:list):
        '''generada en parametros a mano/agregar_regulacion.py'''
        L,A,S,SA,T,aux,R,FD,DTF,DFF,FT=y
        DimFF,DisFF,DimTF,DisTF,kFAc,kTAc,kFLc,kTLc = params
        lfy_ODE  = (DFF/(kFLc*(1+(DTF/kTLc))+DFF))*(((aux+bauxL)/(aux+bauxL+kauxL))*(kLL**n/(L**n+kLL**n))*(S**n/(S**n+kSL**n))+(A**n/(A**n+kAL**n)))-deg*L
        ap1_ODE  = (DFF/(kFAc*(1+(DTF/kTAc))+DFF))*((L**n/(L**n+kLA**n))+(S**n/(S**n+kSA**n)))+((-kDimAP1SAS*A*S)+(kDisAP1SAS*SA))-deg*A
        sas_ODE  = (kAS**n/(A**n+kAS**n))*((DFF/(DFF+kFS))+(S**n/(S**n+kSS**n)))+((-kDimAP1SAS*A*S)+(kDisAP1SAS*SA))-deg*S
        tfl1_ODE  = ((S**n/(S**n+kST**n))*((kRT/(kRT+R))*(kSAT**n/(SA**n+kSAT**n))))+(L**n/(L**n+kLT**n))+((-DimTF*T*FD)+(DisTF*DTF))-deg*T
        fd_ODE = (-DimTF*T*FD)+(DisTF*DTF)+(DisFF*DFF)-(DimFF*FT*FD)+(deg*DTF)
        dtf_ODE = (DimTF*T*FD)-(DisTF*DTF)-(deg*DTF)
        dff_ODE = (DimFF*FT*FD)-(DisFF*DFF)
        return np.array([lfy_ODE,
                ap1_ODE,
                sas_ODE,
                ((kDimAP1SAS*A*S)-(kDisAP1SAS*SA))-deg*SA, # SA
                tfl1_ODE,
                -deg*aux,
                -deg*R,
                fd_ODE,
                dtf_ODE,
                dff_ODE,
                (DisFF*DFF)-(DimFF*FT*FD) # FT
                ])
    
    # exit if the parameters are out of bounds
    if min(theta) <= 0:
        return -np.inf
    
    ln_likelihood = 0
    for ft in Fs:
        for aux_i in initial_auxin:
            r=aux_i
            X0 = np.array([lfy,ap1,sax,dsa,tfl1,aux_i,r,fd,dtf,dff,ft])
            sol = solve_ivp(dX_dt_comp,t,X0,method='LSODA',args=(params,))
            # Salir si cualquier gen se encuentra negativo en cualquier momento:
            for var in sol.y:
                if any(k<0 for k in var):
                    return -np.inf
            # guardar los puntos finales que me interesan:
            L_f = sol.y[0][-1]
            A_f = sol.y[1][-1]
            S_f = sol.y[2][-1]
            T_f = sol.y[8][-1] # DTF
            # -------------------------- Calcular el likelihood -------------------------- #
            if ft == 0: # VEG
                for g in [0,1,2,8]:
                    ln_likelihood += (-0.5 * (sol.y[g][-1])**2 / 0.1**2) # Todos en cero
            elif ft >= 8 and aux_i == 10: # FLOR
                ln_likelihood += (-0.5 * (delta(L_f,S_f))**2 / 0.1**2) # L > S
                ln_likelihood += (-0.5 * (delta(L_f,T_f))**2 / 0.1**2) # L > T
                ln_likelihood += (-0.5 * (delta(A_f,S_f))**2 / 0.1**2) # A > S
                ln_likelihood += (-0.5 * (delta(A_f,T_f))**2 / 0.1**2) # A > T
            else: # INF
                ln_likelihood += (-0.5 * (delta(S_f,L_f))**2 / 0.1**2) # S > L
                ln_likelihood += (-0.5 * (delta(S_f,A_f))**2 / 0.1**2) # S > A
                ln_likelihood += (-0.5 * (delta(T_f,L_f))**2 / 0.1**2) # T > L
                ln_likelihood += (-0.5 * (delta(T_f,A_f))**2 / 0.1**2) # T > A
    return ln_likelihood


# --------------------------- Constantes para MCMC --------------------------- #
moves = [
	(emcee.moves.StretchMove(), 0.1),
	(emcee.moves.WalkMove(), 0.1),
	(emcee.moves.DEMove(), 0.8)
]

# --------------------------- Parámetros iniciales --------------------------- #
# valores desde donde MCMC inicia la búsqueda
# Iniciar con los valores que ya definí a mano
# pero variando un orden de magnitud un parámetro a la vez

param_names = 'DimFF','DisFF','DimTF','DisTF','kFAc','kTAc','kFLc','kTLc'

# Parámetros de dímeros DFF y DTF:
params = []
with open('combinaciones_buenas_1000_favs.txt', 'r') as file:
    for i in file:
        params.append([float(k) for k in i.split(',')])
params = np.array(params)
params_dimers = params.mean(axis=0)
DimFF,DisFF,DimTF,DisTF = params_dimers

# Parámetros de Competitive inhibitions
    # ver notas coliflor/regulaciones por competencia.../segundo y tercer intento
kFAc = 1
kTAc = 1
kFLc=0.1
kTLc=0.01

# Generar una distribución normal de los valores de parámetros ya definidos
# para definir así los valores iniciales de las caminatas
p0 = [DimFF,DisFF,DimTF,DisTF,kFAc,kTAc,kFLc,kTLc]
p1 = np.zeros((n_walkers,len(p0)))
for i in range(n_walkers):
    p0_normal = [max(np.random.normal(k,k*alpha), 0) for k in p0] # Pongo el max para evitar negativos
    for j in range(len(p0)):
        p1[i][j] = p0_normal[j]
# NOTAS SOBRE LO QUE ACABA DE PASAR:
    # en p1 se obtienen 100(=n_walkers) listas de parámetros
    # con valores muy cercanos a cada valor de p0, 
    # definidos por una distribución normal con media en p0
    # y desviación estándar (= k*alpha) proporcional al valor del parámetro

# Por cada parámetro generar un array igual a p1 pero cambio dicho parámetro en un orden de magnitud
p2 = []
for p in range(ndim): # para cada parámetro
    pp = np.copy(p1)
    dif = p0[p] * 10 #al parámetro p aumentarle un orden de magnitud
    for w in range(len(pp)): # por cada caminata cambiar el valor de la media de la dist. normal del parámetro p
        pp[w,p] = max(np.random.normal(dif, dif*alpha), 0) # en caminata w cambiar sólo el valor del parámetro p
    p2.append(pp)
p2 = np.asarray(p2)


# ---------------------------------------------------------------------------- #
#                  Correr el MCMC, pero uno por cada parámetro                 #
# ---------------------------------------------------------------------------- #
for cte in range(ndim): # para cada parámetro
    p3 = np.copy(p2[cte]) # el p1, pero con el parámetro cte igual a su valor original + un orden de magnitud

    # --------------------------- Run the MCMC sampler --------------------------- #
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, moves=moves)
    # Run the sampler for nSteps steps
    sampler.run_mcmc(p3, n_steps, progress=True)
    # get the samples
    samples = sampler.get_chain(flat=True)
    ### RECORDAR: sampler.get_chain guarda array por pasos: [1er Walk paso 0, 2nd walk p 0,...,100 walk paso 0, 1er walk paso 1,...] 


    # ---------------------------- Guardar lo obtenido --------------------------- #
    ### Guardar las samples completas
    filename = 'dimers_B' + str(intento) + '_' + param_names[cte] + '.txt'
    open(filename,'w').close() #para limpiarlo
    with open(filename,'a') as f:
        for i in samples:
            f.write(' '.join(str(item) for item in i)+'\n')

    ### Guardar maximum y average
    # Obtener aquellos con max likelihood:
    p_max = samples[np.argmax(sampler.get_log_prob()),:] 
        #p_max: get the sample (from all the walks and their steps) that has the maximum likelihood parameters 
        #getting into account all the parameters

    # Obtener promedio de cada samples para cada parámetro (sin contar warmup):
    warmup = int(0.8 * n_steps * n_walkers)
    p_avg = np.mean(samples[warmup:,:],axis=0)

    # guardar ambos datos:
    p = p_max,p_avg
    filename = 'max_avg_B' + str(intento) + '_' + param_names[cte] + '.txt'
    np.savetxt(filename, p)
