#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:20:39 2017
@author: humbird1

1D Radiation Hydrodynamics Code

Compute areas at cell edges, volumes at cell centers
Specify radiation BC
Specify u or P BC for u eq
Define material props (Cv, Ka, Kt, gamma,)
Define spatial mesh dr
Define Courant factor function (udt/dx) to call at each cell, time step

At each time step:
Calculate dt in each cell, use minimum of all options
Predictor step: 
    Solve for u
    Solve for r
    Solve for rho
    Solve for E
    Solve for e
    Solve for T, P
Corrector step:    
    Solve for u
    Solve for r
    Solve for rho
    Solve for E
    Solve for e
    Solve for T, P
    
Compute energy conservation

Repeat    
"""
import numpy as np

def calc_grid_area_vol(N,Rmax,geometry,Nt):
    r=np.zeros((N+1,Nt))
    r[:,0]=np.linspace(0,Rmax,N+1)
    A=np.zeros((N+1,Nt))
    V=np.zeros((N,Nt))
    for i in range(0,N+1):
        if (geometry=='Slab'):
            A[i,0]=1.0
        if (geometry=='Cylinder'):
            A[i,0]=2.0*np.pi*r[i,0]
        if (geometry=='Sphere'):
            A[i,0]=4.0*np.pi*r[i,0]**2
    for i in range(0,N):
        if (geometry=='Slab'):
            V[i,0]=r[i+1,0]-r[i,0]
        if (geometry=='Cylinder'):
            V[i,0]=np.pi*(r[i+1,0]**2-r[i,0]**2)    
        if (geometry=='Slab'):
            V[i,0]=4.0*np.pi*(r[i+1,0]**3-r[i,0]**3)
    return(r,A,V)
      
def get_dt(u,dx,P,gamma,rho,k):
    cs=np.zeros(len(P))
    for i in range(0,len(P)): cs[i]=np.sqrt(gamma*P[i,k]/rho[i,k])
    Fc=0.5  #Courant factor
    dt1=np.zeros(len(u))
    for i in range(0,len(u)):dt1[i]=Fc*dx/u[i,k]
    dt2=np.zeros(len(u))
    for i in range(0,len(P)): dt2[i]=dx*Fc/cs[i]
    dtmax=0.1
    dt_choices=np.hstack((dt1,dt2,dtmax))
    dt=np.min(dt_choices)
    return(dt) 
    
def get_abs_opacity(T,k):
    k1=1;k2=1;k3=1;
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*T[i,k]**n+k3)
    return(kappa)

def get_tot_opacity(T,k):
    k1=1;k2=1;k3=1;
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*T[i,k]**n+k3)
    return(kappa)
    
def get_velocity(u,dt_prev,dt,m,A,P,RadE,k):  #i is i+1/2, k is k+1/2
    for i in range(0,len(u)):
        u[i,k]=u[i,k-1]-0.5*(dt_prev+dt)*(A[i,k-1]/m[i])*(P[i+1,k-1]+(1./3.)*RadE[i+1,k-1]-P[i,k-1]-(1./3.)*RadE[i,k-1])    
    return(u)
    
def get_coords(r,u,dt,dt_prev,k):
    for i in range(0,len(r)):
        r[i,k]=r[i,k-1]+0.5*(u[i,k-1]+u[i,k])*0.5*(dt_prev+dt)
    return(r)

def get_area(A,k,r,geometry):
    for i in range(0,len(A)):
        if (geometry=='Slab'):
            A[i,k]=1.0
        if (geometry=='Cylinder'):
            A[i,k]=2.0*np.pi*r[i,k]
        if (geometry=='Sphere'):
            A[i,k]=4.0*np.pi*r[i,k]**2
    return(A)
    
def get_vol(V,k,r,geometry):   
    for i in range(0,len(V)):
        if (geometry=='Slab'):
            V[i,k]=r[i+1,k],-r[i,k]
        if (geometry=='Cylinder'):
            V[i,k]=np.pi*(r[i+1,k]**2-r[i,k]**2)    
        if (geometry=='Slab'):
            V[i,k]=4.0*np.pi*(r[i+1,k]**3-r[i,k]**3)
    return(V)
    
def get_mass_dens(rho,m,V,k):
    for i in range(0,len(m)):
        rho[i,k]=m[i]/V[i,k]
    return(rho)
    
def predictor_rad_E(RadE,A,m,r,u,rho,T,P,Cv,Ka,Kt,dt_prev,dt,k):
    #need to specify BCs
    a=1.0
    c=1.0
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity(T,k)
    Tp=T.copy()
    Tm=T.copy()
    for i in range(1,len(T)):
        Tp[i]=((T[i,k-1]**4+T[i+1,k-1]**4)/2.0)**0.25
        Tm[i]=((T[i,k-1]**4+T[i-1,k-1]**4)/2.0)**0.25
    kappaTp=get_tot_opacity(Tp,k)
    kappaTm=get_tot_opacity(Tm,k)
    nu=np.zeros(len(r)-2)
    xi=np.zeros(len(r)-2)
    F0p=np.zeros(len(r)-2)
    F0m=np.zeros(len(r)-2)
    for i in range(1,len(r)-1):
        nu[i]=(dtk*kappaA[i]*c*2*a*T[i,k-1]**3)/(Cv[i]+dtk*kappaA[i]*c*2*a*T[i,k-1]**3)
        xi[i]=-P[i,k-1]*(A[i+1,k]*0.5*(u[i+1,k-1]+u[i+1,k])- A[i,k]*0.5*(u[i,k-1]+u[i,k]))   
        F0p[i]=-2.0*c/(3.0*(rho[i,k]*(r[i,k]-r[i-1,k])*kappaTp[i]+rho[i+1,k]*(r[i+1,k]-r[i,k])*kappaTp[i+1])) #not sure about opacity eval
        F0m[i]=-2.0*c/(3.0*(rho[i,k]*(r[i,k]-r[i-1,k])*kappaTm[i]+rho[i+1,k]*(r[i+1,k]-r[i,k])*kappaTm[i+1])) #not sure about opacity eval
    C=np.zeros((len(RadE),3))
    Q=np.zeros(len(RadE))
    for i in range(1,len(RadE)-1):
        C[i,i-1]=0.5*F0m[i]*0.5*(A[i,k-1]+A[i,k])
        C[i,i]=m[i]/(dtk*rho[i,k])+0.5*m[i]*kappaA[i]*c*(1.0-nu[i])-0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]
        C[i,i+1]=0.5*F0p[i]*0.5*(A[i+1,k-1]+A[i+1,k])
        Q[i]=nu[i]*xi[i]+m[i]*kappaA[i]*c*(1.0-nu[i])*(a*T[i,k-1]**4-0.5*RadE[i,k-1])
        -(1./3.0)*RadE[i,k-1]*(A[i+1,k-1]*0.5*(u[i+1,k-1]+u[i+1,k])-A[i,k-1]*0.5*(u[i,k-1]+u[i,k]))
        +0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]*(RadE[i,k-1]-RadE[i-1,k-1])
        -0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]*(RadE[i+1,k-1]-RadE[i,k-1])+m[i]*RadE[i,k-1]/(dtk*rho[i,k-1])
    RadE[:,k]=np.linalg.solve(C,Q)
    return(RadE)         
    
def predictor_internale(e,RadE,T,P,A,u,dt_prev,dt,k,Cv,m)    :
    a=1.0
    c=1.0
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity(T,k)    
    xi=np.zeros(len(r)-2)
    for i in range(1,len(r)-1):
        xi[i]=-P[i,k-1]*(A[i+1,k]*0.5*(u[i+1,k-1]+u[i+1,k])- A[i,k]*0.5*(u[i,k-1]+u[i,k]))     
    for i in range(0,len(e)):
        e[i,k]=e[i,k-1]+(dtk*Cv[i]*(m[i]*c*kappaA[i]*(0.5(RadE[i,k]+RadE[i,k-1])-a*T[i,k-1]**4)+xi[i]))/(m[i]*Cv[i]+dtk*m[i]*kappaA[i]*c*2*a*T[i,k-1]**3)
    return(e)    
        
def get_T(T,e,Cv,k):    
    for i in range(0,len(T)):
        T[i,k]=e[i,k]/Cv[i]
    return(T)
    
def get_P(P,e,gamma,rho,k):
    for i in range(0,len(P)):
        P[i,k]=(gamma-1)*rho[i,k]*e[i,k]
    return(P)
    
def get_Apk(A,k):
    Apk=np.zeros(len(A))
    for i in range(0,len(A)):
        Apk[i]=0.5*(A[i,k]+A[i,k-1])
    return(Apk)
    
def get_Ppk(P,k):
    Ppk=np.zeros(len(P))
    for i in range(0,len(P)):
        Ppk[i]=0.5*(P[i,k]+P[i,k-1])
    return(Ppk)

def get_RadEpk(RadE,k):
    RadEpk=np.zeros(len(RadE))
    for i in range(0,len(RadE)):
        RadEpk[i]=0.5*(RadE[i,k]+RadE[i,k-1])
    return(RadEpk)

#corrector u step, call u function sending Apk, Ppk, RadEpk    
#corrector r, rho steps use the same function as predictor step     
    

def corrector_rad_E(RadE,A,m,r,u,rho,T,P,Cv,Ka,Kt,dt_prev,dt,k):
    #need to specify BCs
    a=1.0
    c=1.0
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity(T,k)
    Tp=T.copy()
    Tm=T.copy()
    for i in range(1,len(T)):
        Tp[i]=((T[i,k-1]**4+T[i+1,k]**4)/2.0)**0.25  #maybe change these, not sure...
        Tm[i]=((T[i,k-1]**4+T[i-1,k]**4)/2.0)**0.25
    kappaTp=get_tot_opacity(Tp,k)
    kappaTm=get_tot_opacity(Tm,k)
    nu=np.zeros(len(r)-2)
    xi=np.zeros(len(r)-2)
    F0p=np.zeros(len(r)-2)
    F0m=np.zeros(len(r)-2)
    for i in range(1,len(r)-1):
        nu[i]=(dtk*kappaA[i]*c*2*a*T[i,k-1]**3)/(Cv[i]+dtk*kappaA[i]*c*2*a*T[i,k]**3)
        xi[i]=-P[i,k-1]*(A[i+1,k]*0.5*(u[i+1,k-1]+u[i+1,k])- A[i,k]*0.5*(u[i,k-1]+u[i,k]))   
        F0p[i]=-2.0*c/(3.0*(rho[i,k]*(r[i,k]-r[i-1,k])*kappaTp[i]+rho[i+1,k]*(r[i+1,k]-r[i,k])*kappaTp[i+1])) #not sure about opacity eval
        F0m[i]=-2.0*c/(3.0*(rho[i,k]*(r[i,k]-r[i-1,k])*kappaTm[i]+rho[i+1,k]*(r[i+1,k]-r[i,k])*kappaTm[i+1])) #not sure about opacity eval
    C=np.zeros((len(RadE),3))
    Q=np.zeros(len(RadE))
    for i in range(1,len(RadE)-1):
        C[i,i-1]=0.5*F0m[i]*0.5*(A[i,k-1]+A[i,k])
        C[i,i]=m[i]/(dtk*rho[i,k])+0.5*m[i]*kappaA[i]*c*(1.0-nu[i])-0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]
        C[i,i+1]=0.5*F0p[i]*0.5*(A[i+1,k-1]+A[i+1,k])
        Q[i]=nu[i]*xi[i]+m[i]*kappaA[i]*c*(1.0-nu[i])*(a*T[i,k-1]**4-0.5*RadE[i,k-1])
        -(1./3.0)*RadE[i,k-1]*(A[i+1,k-1]*0.5*(u[i+1,k-1]+u[i+1,k])-A[i,k-1]*0.5*(u[i,k-1]+u[i,k]))
        +0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]*(RadE[i,k-1]-RadE[i-1,k-1])
        -0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]*(RadE[i+1,k-1]-RadE[i,k-1])+m[i]*RadE[i,k-1]/(dtk*rho[i,k-1])
    RadE[:,k]=np.linalg.solve(C,Q)
    return(RadE) 
    
    