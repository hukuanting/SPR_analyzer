import os
import datetime
import csv
import csv
import sys
import matplotlib.pyplot as plt
import numpy as num
#import matplotlib.pyplot as plt
import math
import cmath
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def phase_unwarp (phi):
    shift=0
    for i in range(0,len(phi)-1,1):
        delta=phi[i]-phi[i+1]
        threshold=1.5*math.pi
        if delta>threshold:
              shift=2*math.pi
        if delta<-threshold:
              shift=-2*math.pi
        phi[i+1]=phi[i+1]+shift
    return phi      
     
def make_2x2_array(a, b, c, d, dtype=complex):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = num.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

def Fresnel(wl, theta_i_scalar, h, n_Stack, pol):
    theta=[]
    Fr=[]
    Ft=[]
    delta=[]
    theta.append(theta_i_scalar*math.pi/180)
    length=len(n_Stack)-1
    for i in range(0,length,1):
        Complex_theta=cmath.asin((n_Stack[i]/n_Stack[i+1])*cmath.sin(theta[i]))
        Real_Part=Complex_theta.real
        Imag_Part=-1*abs(Complex_theta.imag)
        next_theta=complex(Real_Part,Imag_Part)
        theta.append(next_theta)
          
    if pol==0:
#        for a=1:length(n)-1 
        for a in range(0,length,1):
            Fr.append((n_Stack[a]*cmath.cos(theta[a])-n_Stack[a+1]*cmath.cos(theta[a+1]))/(n_Stack[a]*cmath.cos(theta[a])+n_Stack[a+1]*cmath.cos(theta[a+1])))
            Ft.append(2*n_Stack[a]*cmath.cos(theta[a])/(n_Stack[a]*cmath.cos(theta[a])+n_Stack[a+1]*cmath.cos(theta[a+1])))
#        Fr(a)=(n(a)*cos(theta(a))-n(a+1)*cos(theta(a+1)))/(n(a)*cos(theta(a))+n(a+1)*cos(theta(a+1)));
#        Ft(a)=2*n(a)*cos(theta(a))/(n(a)*cos(theta(a))+n(a+1)*cos(theta(a+1)));        
             
        
         
    if pol==1: 
        for a in range(0,length,1):
            Fr.append((n_Stack[a]*cmath.cos(theta[a+1])-n_Stack[a+1]*cmath.cos(theta[a]))/(n_Stack[a]*cmath.cos(theta[a+1])+n_Stack[a+1]*cmath.cos(theta[a])))
            Ft.append(2*n_Stack[a]*cmath.cos(theta[a])/(n_Stack[a]*cmath.cos(theta[a+1])+n_Stack[a+1]*cmath.cos(theta[a])))
    
#phase shift factor
    for a in range(0,(length-1),1):
        delta.append(2*math.pi*h[a+1]*cmath.cos(theta[a+1])*n_Stack[a+1]/(wl))
#   
#    print ("-----Delta function---")
#    print (delta)
#    print ("end of delta function")       
        
    M=make_2x2_array(1,0,0,1,complex)#Jones vector 
    for a in range(0,(length-1),1):
#        print(a)
#        print(Fr)
        Mr=make_2x2_array(1,Fr[a],Fr[a],1,complex) 
        Mi=make_2x2_array(cmath.exp(complex(0,-1)*delta[a]),0,0,cmath.exp(complex(0,1)*delta[a]),complex)
        M=M*(1/Ft[a])
        M=num.dot(M,Mr)
        M=num.dot(M,Mi)
#        
#    print ("-----Jones Marticx---")
#    print (M)
#    print ("end of Jones Matrix")   
    Mf=make_2x2_array(1,Fr[length-1],Fr[length-1],1,complex)
    M=M*(1/Ft[length-1])
    M=num.dot(M,Mf)
   
    
    Frtot=M[1][0]/M[0][0]
    Fttot=1/M[0][0]
    #print Frtot
    
    if length==2:
       Frtot=Fr[0]
       Fttot=Ft[0]
       
    FR=(abs(Frtot))**2
    #print FR
    FT_complex_U=n_Stack[length]*cmath.cos(theta[length])
    FT_complex=n_Stack[0]*cmath.cos(theta[0])
    FT_complex_U=FT_complex_U.real
    FT_complex=FT_complex.real
    FT=((abs(Fttot))**2)*FT_complex_U/FT_complex;
    FA=1-FR-FT
    r=Frtot
    return r,FR,FT,FA


#from here on, we can see the main body of the code.
#You may alter the simulation condition as you will.

 
wavelength=850
thetai=list(num.arange(45,67.5,0.001))
n_thetai=len(thetai)
nAu=complex(0.118,5.51) 
nCr=complex(3.24,3.29)#850 nm
pol=1
nDNA=1.42
hDNA=3

n_composite=[1.71,nCr, nAu,1.3460]
#nCr=complex(3.0,3.4035)#730 nm
#nCr=complex(3.17,3.3)#617 nm
#nAu=complex(0.134,4.32)#730 nm
#nAu=complex(0.21,3.27)#617 nm

r=[]
RA=[]
# n_composite=[1.71,nCr, nAu, hDNA, 1.38]
h=[float('nan'),0.1, 50, hDNA, float('nan')]
for i in range(0,n_thetai,1):
    value=Fresnel(wavelength,thetai[i],h,n_composite,pol)
    r.append(value[0])
    RA.append(value[1])

Xdata=thetai
Ydata=r
plt.plot(Xdata,Ydata,marker="o",markersize=1,color='r',label='phase')
plt.plot(Xdata,RA,marker="o",markersize=1,color='#53c7c9',label='intensity')
plt.xlim(50,60)


plt.xlabel('incident angle (degree)')
plt.ylabel('phase (rad)')
plt.tight_layout()

plt.legend()
plt.grid()
plt.show()
#