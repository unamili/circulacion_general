import numpy as np
import time
import matplotlib.pyplot as plt
from Cargar import cargar
import matplotlib.animation as animation
import math
import time 
import os 

#%% Cargamos las salidas del modelo
#Usamos la función "cargar" que está dentro del módulo "Cargar"

Lx=4000 #tamaño de la cuenca (dir x)
Ly=2000 #tamaño de la cuenca (dir y)
nx=200 #puntos de grilla (dir x)
ny=100 #puntos de grilla (dir y)

dir_salida1='/Users/mini/Documents/Circulación/Océano/out_tmp1/'
psi_temp1,vort_temp1,psiF1,vortF1,QG_diag1,QG_curlw1,X1,Y1,dx1,dy1=cargar(dir_salida1,Lx,Ly,nx,ny)

dir_salida2='/Users/mini/Documents/Circulación/Océano/out_tmp2/'
psi_temp2,vort_temp2,psiF2,vortF2,QG_diag2,QG_curlw2,X2,Y2,dx2,dy2=cargar(dir_salida2,Lx,Ly,nx,ny)

dir_salida3='/Users/mini/Documents/Circulación/Océano/out_tmp3/'
psi_temp3,vort_temp3,psiF3,vortF3,QG_diag3,QG_curlw3,X3,Y3,dx3,dy3=cargar(dir_salida3,Lx,Ly,nx,ny)

#en la matriz QG_diag tengo para todos los pasos temporales la función corriente (segunda columna), la vorticidad  (tercera) y la energía cinética (cuarta)

#%%Estabilización del modelo

#Quiero graficar la energía cinética para ver cuándo el modelo se estabiliza (me interesa el estado estacionario)

plt.figure()
plt.plot(QG_diag1[:,3], label='K1')
plt.plot(QG_diag2[:,3], label='K2')
plt.plot(QG_diag3[:,3], label='k3')
plt.ylabel('')
plt.xlabel('Pasos temporales')
plt.grid()
plt.title('Energia Cinética')
plt.legend()
plt.savefig("Energia cinetica.png")


#%%Dimensionalización de las variables psi y vort salidas del modelo

#Defino parámetros de utilidad con valores típicos del problema
B=10**-11 #beta
D=3000 #priofundidad (m)
Rho=1026 #densidad
T_s=0.4 #stress del viento
L=4000000 #escala tipica de la longitud de la cuenca en metros
U=2*math.pi*T_s/(Rho*D*L*B) #escala tipica de la velocidad 

#Dimensionalizamos:
psiF1_dim=psiF1*L*U #función corriente
psiF2_dim=psiF2*L*U
psiF3_dim=psiF3*L*U

vortF1_dim=vortF1*U/L #vorticidad
vortF2_dim=vortF2*U/L
vortF3_dim=vortF3*U/L

#%%Funciones corrientes de cada salida

xx, yy = np.meshgrid(X1, Y1) #Me genero la grilla para luego poder graficar. El xx tiene la información en la dir x (o sea va desde 0 a 4000)
#la grilla va a ser la misma para las 3 simulaciones

#Defino escalas para que los graficos sean comparables
levels_psi=np.arange(-640000,10000,10000) #vector de -640000 a 0 cada 10000

fig4=plt.figure()
plt.contour(xx,yy,psiF1_dim, colors='k')
plt.contourf(xx,yy,psiF1_dim, levels_psi)
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Funcion Corriente 1 (m^2/s)")   #nombre titulo
plt.colorbar()
plt.savefig("FuncionCorriente_1.png")

fig5=plt.figure()
plt.contour(xx,yy,psiF2_dim, colors='k')
plt.contourf(xx,yy,psiF2_dim,levels_psi)
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Funcion Corriente 2 (m^2/s)")   #nombre titulo
plt.colorbar()    #agrego la barra de colores
plt.savefig("FuncionCorriente_2.png")

fig6=plt.figure()
plt.contour(xx,yy,psiF3_dim, colors='k')
plt.contourf(xx,yy,psiF3_dim,levels_psi)
#plt.contour(xx,yy,psiF3_dim, colors='k')
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Funcion Corriente 3 (m^2/s)")   #nombre titulo
plt.colorbar()    #agrego la barra de colores
plt.savefig("FuncionCorriente_3.png")

#%%Transporte meridional My

#Para obtener My saco el diferencial de psi
dif1=np.diff(psiF1_dim,n=1,axis=1)  #para la simulación 1
My1=dif1*D/(10**6) #Lo divido por 10^6 para obtener el transporte en Sv
#plt.imshow(My1)  para ver la variable que acabo de definir

dif2=np.diff(psiF2_dim,n=1,axis=1)  #para la simulación 2
My2=dif2*D/(10**6) #Lo divido por 10^6 para obtener el transporte en Sv

dif3=np.diff(psiF3_dim,n=1,axis=1)  #para la simulación 3
My3=dif3*D/(10**6) #Lo divido por 10^6 para obtener el transporte en Sv

#Me genero otra grilla para poder graficar My porque la dim es 99x200
xx1, yy1 = np.meshgrid(X1[1:], Y1)

#Defino escalas para que los graficos sean comparables
levels_my=np.arange(-500,15,5) #vector de -175 a 25 cada 5

#Uso la misma grilla para los 3
fig7=plt.figure()
#plt.contour(xx1[:,:20],yy1[:,:20],My1[:,:20], colors ='k')
plt.contourf(xx1 [:,:20] , yy1[:,:20] , My1[:,:20] , levels_my )
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Transporte 1 (Sv)")   #nombre titulo
plt.colorbar()   #agrego la barra de colores
plt.savefig("Transporte_1.png")

fig8=plt.figure()
#plt.contour(xx1[:,:20],yy1[:,:20],My2[:,:20], colors ='k')
plt.contourf(xx1[:,:20],yy1[:,:20],My2[:,:20],levels_my)
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Transporte 2 (Sv)")   #nombre titulo
plt.colorbar()   #agrego la barra de colores
plt.savefig("Transporte_2.png")

fig9=plt.figure()
#plt.contour(xx1[:,:20],yy1[:,:20],My3[:,:20], colors ='k')
plt.contourf(xx1[:,:20],yy1[:,:20],My3[:,:20],levels_my)
plt.xlabel("W <= (km) => E")   #nombre eje x
plt.ylabel ("S <= (km) => N")   #nombre eje y
plt.title("Transporte 3 (Sv)")   #nombre titulo
plt.colorbar()   #agrego la barra de colores
plt.savefig("Transporte_3.png")


#%% Transporte CENTRO DE LA CUENCA - en las distintas longitudes 

My_central1=My1[50,:] #Me quedo con la fila 50 de las matrices del transporte My
My_central2=My2[50,:]
My_central3=My3[50,:]

#aca me estoy armando los puntos de coordenada (en longitud) correspondiente a la cuenca
seg = L/200   #cada segmento mide esto (es como el dx pero entero)
segmentos = np.arange(seg, L , seg)/1000 #con esto me armo una lisa que arranca en L/200 (que es 20) y ue llega el anteultimo punto de grilla. osea que tengo 198 puntos de grilla en total

#   armado de figura
#ax1=plt.subplot(311)  #es otra forma de hacer figuras. creo una figura vacia (o eso entiendo)

plt.figure()
plt.plot(segmentos, My_central1, label='K1')
plt.plot(segmentos, My_central2, label='K2')
plt.plot(segmentos, My_central3, label='K3')
plt.xlabel("W <= (km) => E  ")
plt.ylabel("Sv = 10^6 m^3/s")
plt.grid()
plt.title('Transporte Meridional en el medio de la cuenca ')
plt.legend()
plt.xlim([-0.5,1000])
plt.savefig("TransporteCentral.png")

#%% Vorticidad CENTRO DE LA CUENCA - en las distintas longitudes
 
Vort_central1=vortF1_dim[50,:] #Me quedo con la fila 50 de las matrices del transporte My
Vort_central2=vortF2_dim[50,:]
Vort_central3=vortF3_dim[50,:]

plt.figure()
plt.plot(segmentos, Vort_central1[1:], label='K1')
plt.plot(segmentos, Vort_central2[1:], label='K2')
plt.plot(segmentos, Vort_central3[1:], label='K3')
plt.xlabel("W <= (km) => E  ")
plt.ylabel("Vorticidad (1/s)")
plt.grid()
plt.title('Vorticidad en el medio de la cuenca ')
plt.legend()
plt.xlim([-0.5,1000])
plt.savefig("VorticidadCentral.png")

#%%Transporte de la CBO

#corrida 1
i=0  
transp1=0
while My_central1[i]<0 and i<len(My_central1): #considero CBO cuando el My es negativo
    transp1=transp1+My_central1[i]
    i=i+1
    
CBO_ancho1=i*dx1 #el valor de i es la cantidad de puntos de grilla donde está la CBO y dx1 la distancia en km que hay entre los puntos

#corrida 2
i=0  
transp2=0
while My_central2[i]<0 and i<len(My_central2): 
    transp2=transp2+My_central2[i]
    i=i+1
    
CBO_ancho2=i*dx2 

#corrida 3
i=0  
transp3=0
while My_central3[i]<0 and i<len(My_central3): 
    transp3=transp3+My_central3[i]
    i=i+1
    
CBO_ancho3=i*dx3 


#%% Transporte meridional total
# uso la funcion suma sobre los cortes centrales

My_total_1= sum(My_central1)

My_total_2= sum(My_central2)

My_total_3= sum(My_central3)



#%% ejercicio 3. Usamos la simulacion numero 2

#dif=np.diff(psiF2,n=1,axis=1) #le aplico el operador diferencial (en ex) a psi2
#dif_central= dif[50,:]
#QG_curlw2_central = QG_curlw2[50,:]

#%% 3 con k1
ds=0.05
#primer termino
primer_termino=((np.diff(psiF2,n=1, axis=1)))[50,:] /ds #este es el que tuvo que tirar magia dani con ese ds que viene del .dat (gridstep)
segundo_termino=-QG_curlw2[51,:]
tercer_termino=0.29*vortF2[50,:]
#mjhhj

plt.figure(33)
plt.plot(primer_termino,"r", label='Dif fi')
plt.plot(segundo_termino,"b", label= 'Rotor')
plt.plot(tercer_termino,"g", label= 'Vort')
plt.legend()
plt.xlabel("W <==> E")
plt.ylabel ('')
plt.grid()
plt.legend()
plt.savefig("Ej3.png")


neto = primer_termino -segundo_termino[2] -(-tercer_termino)






