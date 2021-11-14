#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:30:46 2021

@author: espi
"""
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import pandas_datareader as pdr
from datetime import datetime
np.random.seed(6829)

#Fechas de inicio y fin de las Siglas que queramos analizar
#y comparar con caminatas aleatorias
empresa =input('Introduce Siglas Empresa:')
fecha_in=input('Introduce fecha de comienzo,'+
                   'en forma Año/Mes/Día-XXXX/XX/XX:')
fecha_fin=input('Introduce fecha de fin:')
año_i=int(fecha_in[0:4]); mes_i=int(fecha_in[5:7]); dia_i=int(fecha_in[8:10])
año_f=int(fecha_fin[0:4]); mes_f=int(fecha_fin[5:7]); dia_f=int(fecha_fin[8:10])
datos=pdr.get_data_yahoo(symbols=empresa, start=datetime(año_i,mes_i,dia_i)
                         ,end=datetime(año_f,mes_f,dia_f))[['Close']]
dias=len(datos)

#mu=risk-free rate del mercado (inflación).FB->NASDAQ;Iberia->IBEX35
mu=0.25

#Cálculo Volatilidad. Multiplicamos por la raíz cuadrada
#de los días porque numpy no divide el cuadrado de la diferencia
#entre n
datos['Returns']=datos['Close'].pct_change()
vol=datos['Returns'].std()*math.sqrt(dias)
print("Volatility=", str(round(vol,5)*100)+"%")

#Preparar los parámetros
Yo=datos['Close'][0]
T=dias
mu=round(mu,4)
vol=round(vol,4)

N=dias
h=T/N

#Cálculo log-retornos Real
data=datos['Close'].dropna()
log_retornos = np.log(data / data.shift(1))
log_datos=log_retornos.dropna()

#Def plots de las densidades normalizada. i=1 si queremos ajuste T-student
def plot_PDF_T(array,mu,vol,N,i):
    #Calculamos parámetros T-Student
    (gdl, mu_t, sigma_t) = scs.t.fit(array)
    #Print del histograma con ajuste PDF gaussiano con el r y sigma dados
    plt.hist(array, bins=70 , density=True , label='frecuencia', ec='w')
    plt.grid(True)
    plt.xlabel('log-retornos')
    plt.ylabel('frecuencia')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, loc=mu/N, scale=vol / np.sqrt(N)),'r', lw=2.0, 
             label='PDF Normal')
    if i==1:
        plt.plot(x, scs.t.pdf(x, gdl, loc=mu_t , scale=sigma_t ),'k', lw=2.0, 
                 label='PDF T-Student')
    else:
        pass
    plt.legend()
    return gdl, mu_t, sigma_t
#Deberiamos esperar mean * M= r (ratio de interes) y 
#std*sqrt(50)= sigma (volatilidad) para PDF gaussiano
def estadistica(array):
    sta = scs.describe(array)
    print ("%14s %15s" % ('statistic', 'value'))
    print (30 * "-")
    print ("%14s %15.5f" % ('nº datos', sta[0]))
    print ("%14s %15.5f" % ('min', sta[1][0]))
    print ("%14s %15.5f" % ('max', sta[1][1]))
    print ("%14s %15.5f" % ('media', sta[2]))
    print ("%14s %15.5f" % ('desviación', np.sqrt(sta[3])))
    print ("%14s %15.5f" % ('asimetría', sta[4]))
    print ("%14s %15.5f" % ('kurtosis', sta[5]))
    
#Def comparición de cuantiles de los datos con los cuantiles teóricos Gaussianos,
#cuando más se alejen los puntos de la recta menos similitud en los compor-
#tamientos extremos ( las colas).
def cuantil_plot(array):
    sm.qqplot(array, line='s')
    plt.grid(True)
    plt.xlabel('cuantiles teóricos')
    plt.ylabel('cuantiles muestra')
    plt.legend()


#Def test de sesgo y kurtosis 
#(colas gruesas o finas en comparación a una gaussiana)
def test_normalidad(arr):
    print ("Asimetría de los datos       %14.3f" % scs.skew(arr))
    print ("Valor p-value asimetría test %14.3f" % scs.skewtest(arr)[1])
    print ("Kurtosis de los datos        %14.3f" % scs.kurtosis(arr))
    print ("Valor p-value kurtosis test  %14.3f" % scs.kurtosistest(arr)[1])
    print ("Valor p-value normal pdf     %14.3f" % scs.normaltest(arr)[1])

#Def para generar caminatas aleatorias:
def gen_paths(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + 
                                         sigma * np.sqrt(dt) * rand)
    return paths

#Def para generar una caminata
def gen_path(S0, r, sigma, T, M):
    dt = float(T) / M
    path=[]
    path.append(S0)
    for t in range(1, M + 1):
        rand = np.random.standard_normal()
        path.append(path[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand))
    return path

#Def plot n caminatas aleatorias:
def plot_caminos(caminos,n):
    plt.plot(paths[:, :n])
    plt.grid(True)
    plt.xlabel('time steps')
    plt.ylabel('index level')


#Plots datos reales,ponemos 1 al final para que haga T-Student también
plt.figure(0)
gdl, mu_t, sigma_t=plot_PDF_T(log_datos, mu, vol, N,1); plt.title('Densidad log-retorno real')

#Calculamos los Paths; I=N/M para simular 
#una cantidad de datos semejante a los reales
S0 = Yo; r = mu; sigma = vol; T = 1.0; M = 50; I = int(N/M);
paths = gen_paths(S0, r, sigma, T, M, I)
log_returns = np.log(paths[1:] / paths[0:-1])

#Plots datos simulados
plt.figure(1)
plot_PDF_T(log_returns.flatten(), mu, vol, M,0);
plt.title('Densidad log-retorno simulado')

#Plots de los cuantiles
cuantil_plot(log_datos); plt.title('Cuantiles datos reales')
cuantil_plot(log_returns.flatten()); plt.title('Cuantiles simulaciones')

#Plots de datos históricos reales y de una caminata aleatoria
mov_browniano=gen_path(S0,r,sigma,T,N-1)
TIEMPO=np.linspace(0.,1.,N)
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(211)
plt.title('Data_Real-'+ empresa,fontsize=100, loc='center')
plt.grid('true')
plt.ylabel(empresa + ' Stock Price,$',fontsize=100, loc='center')
plt.xlabel(año_i+'/'+mes_i+'/'+dia_i+'-'+año_f+'/'+mes_f+'/'+dia_f)
plt.plot(TIEMPO,data,'-b')
ax2 = fig.add_subplot(212)
plt.title('Brownian Motion',fontsize=100, loc='center')
plt.grid('true')
plt.ylabel(empresa + ' Stock Price,$',fontsize=100, loc='center')
plt.xlabel(año_i+'/'+mes_i+'/'+dia_i+'-'+año_f+'/'+mes_f+'/'+dia_f)
plt.plot(TIEMPO,mov_browniano,'-b')

#Prints estadísticas, analisis de asimetría y kurtosis.
print('\n ----Test de normalidad---- Datos reales ---- \n')
test_normalidad(log_datos)

print('\n ----Test de normalidad---- Datos Simulados ---- \n')
test_normalidad(log_returns.flatten())

print('\n ---- Estadística ---- Datos reales ----\n')
estadistica(log_datos)

print('\n ----Estadística---- Datos Simulados ---- \n')
estadistica(log_returns.flatten())





