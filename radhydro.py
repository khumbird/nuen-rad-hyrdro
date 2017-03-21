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
    for j in range(0,Nt):
        r[:,j]=np.linspace(0,Rmax,N+1)
    A=np.zeros((N+1,Nt))
    V=np.zeros((N,Nt))
    for i in range(0,N+1):
        if (geometry=='Slab'):
            A[i,:]=1.0
        if (geometry=='Cylinder'):
            A[i,:]=2.0*np.pi*r[i,0]
        if (geometry=='Sphere'):
            A[i,:]=4.0*np.pi*r[i,0]**2
    for i in range(0,N):
        if (geometry=='Slab'):
            V[i,:]=r[i+1,0]-r[i,0]
        if (geometry=='Cylinder'):
            V[i,:]=np.pi*(r[i+1,0]**2-r[i,0]**2)    
        if (geometry=='Sphere'):
            V[i,:]=4.0*np.pi*(r[i+1,0]**3-r[i,0]**3)
    return(r,A,V)
      
def get_dt(u,r,P,gamma,rho,k):
    cs=np.zeros(len(P))
    dx=np.zeros(len(P))
    for i in range(0,len(P)): dx[i]=r[i+1,k]-r[i,k]
    for i in range(0,len(P)): cs[i]=np.sqrt(gamma*P[i,k]/rho[i,k])
    Fc=0.5  #Courant factor
    dt1=np.zeros(len(P))
    for i in range(0,len(P)):dt1[i]=Fc*dx[i]/u[i,k]
    dt2=np.zeros(len(P))
    for i in range(0,len(P)): dt2[i]=dx[i]*Fc/cs[i]
    dtmax=0.01
    dt_choices=np.hstack((dt1,dt2,dtmax))
    dt=np.min(dt_choices)
    return(dt) 
    
def get_abs_opacity(T,k,Tbl,Tbr):
    k1=20.0;k2=0.0;k3=1.0;
    #k1=1;k2=1;k3=1;
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*T[i,k-1]**n+k3)
    Tbound=((Tbl**4+T[0,k-1]**4)/2.0)**(1.0/4.0)
    kappa[0]=k1/(k2*Tbound**n+k3)
    Tbound2=((Tbr**4+T[-1,k-1]**4)/2.0)**(1.0/4.0)
    kappa[-1]=k1/(k2*Tbound2*2*n+k3)        
    return(kappa)

def get_tot_opacity(T,k,Tbl,Tbr):
    k1=20.5;k2=0.0;k3=1.0;
    #k1=1;k2=1;k3=1;
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*T[i,k-1]**n+k3)
    Tbound=((Tbl**4+T[0,k-1]**4)/2.0)**(1.0/4.0)
    kappa[0]=k1/(k2*Tbound**n+k3)
    Tbound2=((Tbr**4+T[-1,k-1]**4)/2.0)**(1.0/4.0)
    kappa[-1]=k1/(k2*Tbound2**n+k3)
    return(kappa)
    
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
            V[i,k]=r[i+1,k]-r[i,k]
        if (geometry=='Cylinder'):
            V[i,k]=np.pi*(r[i+1,k]**2-r[i,k]**2)    
        if (geometry=='Sphere'):
            V[i,k]=4.0*np.pi*(r[i+1,k]**3-r[i,k]**3)
    return(V)
    
def get_mass_dens(rho,m,V,k):
    for i in range(0,len(m)):
        rho[i,k]=m[i,k]/V[i,k]
    return(rho)
    
def artif_viscosity(P,gamma,u,r,rho,k):
    cQ=(gamma+1.0)/4.0
    Q=np.zeros(len(P))
    dr=np.zeros(len(P))
    du=np.zeros(len(P))
    Rm=np.zeros(len(P))
    Rp=np.zeros(len(P))
    rhom=np.zeros(len(P))
    rhop=np.zeros(len(P))    
    csm=np.zeros(len(P))
    csp=np.zeros(len(P))       
    for i in range(0,len(P)):
        dr[i]=r[i+1,k]-r[i,k]
        du[i]=u[i+1,k]-u[i,k]
    for i in range(1,len(P)):
        Rm[i]=du[i-1]*dr[i]/(dr[i-1]*u[i,k])
        rhom[i]=(rho[i-1,k]*dr[i-1]+rho[i,k]*dr[i])/(dr[i-1]+dr[i])
        csm[i]=(np.sqrt(gamma*P[i-1,k]/rho[i-1,k])*dr[i-1]+ np.sqrt(gamma*P[i,k]/rho[i,k])*dr[i])/(dr[i-1]+dr[i])
    for i in range(0,len(P)-1):
        Rp[i]=du[i+1]*dr[i]/(dr[i+1]*u[i,k])
        rhop[i]=(rho[i+1,k]*dr[i+1]+rho[i,k]*dr[i])/(dr[i+1]+dr[i])
        csp[i]=(np.sqrt(gamma*P[i+1,k]/rho[i+1,k])*dr[i+1]+ np.sqrt(gamma*P[i,k]/rho[i,k])*dr[i])/(dr[i+1]+dr[i])
    for i in range(0,len(P)):
        rhobar=2*rhom[i]*rhop[i]/(rhom[i]+rhop[i])
        cs=np.min((csp[i],csm[i]))
        Gam=np.nanmax((0,np.min((1,2*Rm[i],2*Rp[i],0.5*(Rp[i]+Rm[i]) )) ))
        if(du[i]<0):
            Q[i]=(1-Gam)*rhobar*abs(du[i])*(cQ*abs(du[i])+np.sqrt(cQ**2*du[i]**2+cs))
        else: Q[i]=0.0
    return(Q) 

def predictor_boundary_velocity(u,k,m,T,dtk,r,A,P,PbR,PbL,TbR,TbL,RadE,rho,a):
    kappaT=get_tot_opacity(T,k,TbL,TbR)
    RadE12=(3.0*(0.5*(rho[0,k]+rho[0,k-1])*(0.5*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1]))*kappaT[0]*a*TbL**4+4.0*RadE[0,k-1]))/(3.0*(0.5*(rho[0,k]+rho[0,k-1])*(0.5*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1]))*kappaT[0]+4.0))
    RadEN12=(3.0*(0.5*(rho[-1,k]+rho[-1,k-1])*(0.5*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1]))*kappaT[-1]*a*TbR**4+4.0*RadE[-1,k-1]))/(3.0*(0.5*(rho[-1,k]+rho[-1,k-1])*(0.5*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1]))*kappaT[-1]+4.0))    
    uleft=u[0,k-1]-(dtk*A[0,k-1]/(0.5*m[0,k]))*(P[0,k-1]-PbL+(1.0/3.0)*(RadE[0,k-1]-RadE12))    
    uright=u[-1,k-1]-(dtk*A[-1,k-1]/(0.5*m[-1,k]))*(-P[-1,k-1]+PbR+(1.0/3.0)*(RadEN12-RadE[-1,k-1]))   
    bu=[uleft,uright]
    return(bu)
       
def predictor_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma):  #i is i+1/2, k is k+1/2
    dtk=0.5*(dt_prev+dt)    
    bu=predictor_boundary_velocity(u,k,m,T,dtk,r,A,P,PbR,PbL,TbR,TbL,RadE,rho,a)
    print(bu)
    u[0,k]=bu[0]
    u[-1,k]=bu[1]
    QP=artif_viscosity(P,gamma,u,r,rho,k-1)
    for i in range(1,len(u)-1):
        u[i,k]=u[i,k-1]-0.5*(dt_prev+dt)*(A[i,k-1]/(0.5*m[i,k-1]+0.5*m[i-1,k-1]))*(P[i,k-1]+QP[i]+(1./3.)*(RadE[i,k-1]-RadE[i-1,k-1])-P[i-1,k-1]-QP[i-1])    
    return(u)
    
def predictor_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma):
    a=0.01372
    c=299.792
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity(T,k,TbL,TbR)
    Tp=T.copy()
    Tm=T.copy()
    QP=artif_viscosity(P,gamma,u,r,rho,k)
    for i in range(0,len(T)-1):
        Tp[i]=((T[i,k-1]**4+T[i+1,k-1]**4)/2.0)**0.25
        Tm[i]=((T[i,k-1]**4+T[i-1,k-1]**4)/2.0)**0.25
    kappaTp=get_tot_opacity(Tp,k,TbL,TbR)
    kappaTm=get_tot_opacity(Tm,k,TbL,TbR)
    nu=np.zeros(len(T))
    xi=np.zeros(len(T))
    F0p=np.zeros(len(A))
    F0m=np.zeros(len(A))
    for i in range(0,len(T)):
        nu[i]=(dtk*kappaA[i]*c*2*a*T[i,k-1]**3)/(Cv[i]+dtk*kappaA[i]*c*2*a*T[i,k-1]**3)
        xi[i]=-(P[i,k-1]+QP[i])*(A[i+1,k-1]*0.5*(u[i+1,k-1]+u[i+1,k])- A[i,k-1]*0.5*(u[i,k-1]+u[i,k])) 
    for i in range(1,len(A)-1):        
        F0p[i]=-2.0*c/(3.0*(0.5*(rho[i-1,k]+rho[i-1,k-1])*(0.5*(r[i,k]+r[i,k-1]-r[i-1,k]-r[i-1,k-1]))*kappaTp[i]+0.5*(rho[i,k]+rho[i,k-1])*(0.5*(r[i+1,k]+r[i+1,k-1]-r[i,k]-r[i,k-1]))*kappaTp[i])) #not sure about opacity eval
        F0m[i]=F0p[i]#-2.0*c/(3.0*(0.5*(rho[i,k]+rho[i,k-1])*(0.5*(r[i+1,k]+r[i+1,k-1]-r[i,k]-r[i,k-1]))*kappaTm[i]+0.5*(rho[i-1,k]+rho[i-1,k-1])*(0.5*(r[i,k]+r[i,k-1]-r[i-1,k]-r[i-1,k-1]))*kappaTm[i])) #not sure about opacity eval        
    C=np.zeros((len(RadE),len(RadE)))
    Q=np.zeros(len(RadE))
    
    F0p[0]=-2*c/(3*0.5*(rho[0,k]+rho[0,k-1])*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1])*kappaTm[0]+4.0)
    F0m[0]=-2*c/(3*0.5*(rho[0,k]+rho[0,k-1])*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1])*kappaTm[0]+4.0)
    C[0,0]=m[0,k]/(dtk*rho[0,k])+0.5*m[0,k]*kappaA[0]*c*(1.0-nu[0])-0.5*(A[1,k-1]+A[1,k])*0.5*F0p[1]-0.5*(A[0,k-1]+A[0,k])*0.5*F0m[0]
    C[0,1]=0.5*F0p[1]*0.5*(A[1,k-1]+A[1,k])
    Q[0]=nu[0]*xi[0]+m[0,k]*kappaA[0]*c*(1.0-nu[0])*(a*T[0,k-1]**4-0.5*RadE[0,k-1])-(1./3.0)*RadE[0,k-1]*(A[1,k-1]*0.5*(u[1,k-1]+u[1,k])-A[0,k-1]*0.5*(u[0,k-1]+u[0,k]))+ \
         0.5*(A[0,k-1]+A[0,k])*0.5*F0m[0]*(RadE[1,k-1]-a*TbL**4)-0.5*(A[1,k-1]+A[1,k])*0.5*F0p[1]*(RadE[1,k-1]-RadE[0,k-1])+m[0,k]*RadE[0,k-1]/(dtk*rho[0,k-1])

    F0p[-1]=-2*c/(3*0.5*(rho[-1,k]+rho[-1,k-1])*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1])*kappaTm[-1]+4.0)
    F0m[-1]=-2*c/(3*0.5*(rho[-1,k]+rho[-1,k-1])*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1])*kappaTm[-1]+4.0)
    C[-1,-1]=m[-1,k]/(dtk*rho[-1,k])+0.5*m[-1,k]*kappaA[-1]*c*(1.0-nu[-1])-0.5*(A[-2,k-1]+A[-2,k])*0.5*F0m[-2]-0.5*(A[-1,k-1]+A[-1,k])*0.5*F0p[-1]
    C[-1,-2]=0.5*F0m[-2]*0.5*(A[-2,k-1]+A[-2,k])
    Q[-1]=nu[-1]*xi[-1]+m[-1,k]*kappaA[-1]*c*(1.0-nu[-1])*(a*T[-1,k-1]**4-0.5*RadE[-1,k-1])-(1./3.0)*RadE[-1,k-1]*(-A[-2,k-1]*0.5*(u[-2,k-1]+u[-2,k])+A[-1,k-1]*0.5*(u[-1,k-1]+u[-1,k]))+ \
         0.5*(A[-1,k-1]+A[-1,k])*0.5*F0p[-1]*(-a*TbR**4+RadE[-2,k-1])-0.5*(A[-2,k-1]+A[-2,k])*0.5*F0m[-2]*(RadE[-2,k-1]-RadE[-1,k-1])+m[-1,k]*RadE[-1,k-1]/(dtk*rho[-1,k-1])
            
            
#    C[0,0]=1.0;C[-1,-1]=1.0;
#    Q[0]=RadE[0,k-1];Q[-1]=RadE[-1,k-1] #add real BCs
#    Q[0]=0.0;Q[-1]=0.0 #add real BCs    
    for i in range(1,len(RadE)-1):   
        C[i,i-1]=0.5*F0m[i]*0.5*(A[i,k-1]+A[i,k])
        C[i,i]=m[i,k]/(dtk*rho[i,k])+0.5*m[i,k]*kappaA[i]*c*(1.0-nu[i])-0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]
        C[i,i+1]=0.5*F0p[i]*0.5*(A[i+1,k-1]+A[i+1,k])
    for i in range(1,len(RadE)-1):   
        Q[i]=nu[i]*xi[i]+m[i,k]*kappaA[i]*c*(1.0-nu[i])*(a*T[i,k-1]**4-0.5*RadE[i,k-1])-(1./3.0)*RadE[i,k-1]*(A[i+1,k-1]*0.5*(u[i+1,k-1]+u[i+1,k])-A[i,k-1]*0.5*(u[i,k-1]+u[i,k])) \
            +0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]*(RadE[i,k-1]-RadE[i-1,k-1])-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]*(RadE[i+1,k-1]-RadE[i,k-1])+m[i,k]*RadE[i,k-1]/(dtk*rho[i,k-1])
    RadE[:,k]=np.linalg.solve(C,Q)
    return(RadE)         
    
def predictor_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho):
    a=0.01372
    c=299.792
    QP=artif_viscosity(P,gamma,u,r,rho,k)
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity(T,k,TbL,TbR)    
    xi=np.zeros(len(P))
    for i in range(0,len(P)):
        xi[i]=-(P[i,k-1]+QP[i])*(A[i+1,k-1]*0.5*(u[i+1,k-1]+u[i+1,k])- A[i,k-1]*0.5*(u[i,k-1]+u[i,k]))   
    for i in range(0,len(e)):
        e[i,k]=e[i,k-1]+(dtk*Cv[i]*(m[i,k]*c*kappaA[i]*(0.5*(RadE[i,k]+RadE[i,k-1])-a*T[i,k-1]**4)+xi[i]))/(m[i,k]*Cv[i]+dtk*m[i,k]*kappaA[i]*c*2*a*T[i,k-1]**3)
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
def get_abs_opacity_corrector(T,k,Tbl,Tbr,Tp):
    k1=20.0;k2=0.0;k3=1.0;
    #k1=1;k2=1;k3=1;    
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*(0.5*(T[i,k]**n+Tp[i,k]**n))+k3)
    Tbound=((Tbl**4+T[0,1]**4)/2.0)**(1.0/4.0)
    kappa[0]=k1/(k2*Tbound**n+k3)
    Tbound2=((Tbr**4+T[-1,k]**4)/2.0)**(1.0/4.0)
    kappa[-1]=k1/(k2*Tbound2**n+k3)
    return(kappa)    

def get_tot_opacity_corrector(T,k,Tbl,Tbr,Tp):
    k1=20.5;k2=0.0;k3=1.0;
    #k1=1;k2=1;k3=1;    
    n=1
    kappa=np.zeros(len(T))
    for i in range(0,len(T)): kappa[i]=k1/(k2*(0.5*(T[i,k]**n+Tp[i,k]**n))+k3)
    Tbound=((Tbl**4+T[0,k]**4)/2.0)**(1.0/4.0)
    kappa[0]=k1/(k2*Tbound**n+k3)
    Tbound2=((Tbr**4+T[-1,k]**4)/2.0)**(1.0/4.0)
    kappa[-1]=k1/(k2*Tbound2**n+k3)
    return(kappa)       

def corrector_boundary_velocity(u,k,m,T,dtk,r,A,P,PbR,PbL,TbR,TbL,RadE,rho,Tp):
    a=0.01372
    kappaT=get_tot_opacity(T,k,TbL,TbR)
    RadE12=(3*rho[0,k]*(r[1,k]-r[0,k])*kappaT[0]*a*TbL**4+4*RadE[0,k])/(3*rho[0,k]*(r[1,k]-r[0,k])*kappaT[0]+4)
    RadEN12=(3*rho[-1,k]*(r[-1,k]-r[-2,k])*kappaT[-1]*a*TbR**4+4*RadE[-1,k])/(3*rho[-1,k]*(r[-1,k]-r[-2,k])*kappaT[-1]+4)
    uleft=u[0,k]-(dtk*A[0,k]/(0.5*m[0,k]))*(P[0,k]-PbL+(1./3.)*(RadE[0,k]-RadE12))    
    uright=u[-1,k]-(dtk*A[-1,k]/(0.5*m[-1,k]))*(-P[-1,k]+PbR+(1./3.)*(-RadE[-1,k]+RadEN12)) 
    bu=[uleft,uright]
    print(bu)
    return(bu)
      
def corrector_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma,Tp):  #i is i+1/2, k is k+1/2
    Ppk=get_Ppk(P,k)
    Apk=get_Apk(A,k)   
    RadEpk=get_RadEpk(RadE,k)
    dtk=0.5*(dt_prev+dt)    
    QPk=artif_viscosity(P,gamma,u,r,rho,k)
    QPkm1=artif_viscosity(P,gamma,u,r,rho,k-1)
    b=corrector_boundary_velocity(u,k,m,T,dtk,r,A,P,PbR,PbL,TbR,TbL,RadE,rho,Tp)
    u[0,k]=b[0];u[-1,k]=b[1]
    print(b)
    for i in range(1,len(u)-1):
        u[i,k]=u[i,k-1]-dtk*(Apk[i]/m[i,k])*((Ppk[i]+QPk[i])+(1./3.)*(RadEpk[i]-RadEpk[i-1])-(Ppk[i-1]+QPkm1[i-1]))       
    return(u)
    
    
def corrector_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma,RadEp,Ap,Tp):
    a=0.01372
    c=299.792
    dtk=0.5*(dt_prev+dt)    
    QP=artif_viscosity(P,gamma,u,r,rho,k)
    kappaT=get_tot_opacity_corrector(T,k,TbL,TbR,Tp)
    kappaA=get_abs_opacity_corrector(T,k,TbL,TbR,Tp)    
    nu=np.zeros(len(P))
    xi=np.zeros(len(P))
    F0p=np.zeros(len(A))
    F0m=np.zeros(len(A))
    Ppk=get_Ppk(Pp,k)
    Apk=get_Apk(Ap,k)
    RadEpk=get_RadEpk(RadEp,k)
    for i in range(0,len(P)):
        nu[i]=(dtk*kappaA[i]*c*2*a*T[i,k]**3)/(Cv[i]+dtk*kappaA[i]*c*2*a*T[i,k]**3)
        xi[i]=-(Ppk[i]+QP[i])*(Apk[i+1]*0.5*(u[i+1,k-1]+u[i+1,k])- Apk[i]*0.5*(u[i,k-1]+u[i,k]))-(m[i,k]/dtk)*(e[i,k]-e[i,k-1])   

    for i in range(1,len(A)-1):        
        F0p[i]=-2.0*c/(3.0*(0.5*(rho[i-1,k]+rho[i-1,k-1])*(0.5*(r[i,k]+r[i,k-1]-r[i-1,k]-r[i-1,k-1]))*kappaT[i-1]+0.5*(rho[i,k]+rho[i,k-1])*(0.5*(r[i+1,k]+r[i+1,k-1]-r[i,k]-r[i,k-1]))*kappaT[i])) #not sure about opacity eval
        F0m[i]=F0p[i]#-2.0*c/(3.0*(0.5*(rho[i,k]+rho[i,k-1])*(0.5*(r[i,k]+r[i,k-1]-r[i-1,k]-r[i-1,k-1]))*kappaT[i]+0.5*(rho[i-1,k]+rho[i-1,k-1])*(0.5*(r[i,k]+r[i,k-1]-r[i-1,k]-r[i-1,k-1]))*kappaT[i-1])) #not sure about opacity eval
    C=np.zeros((len(RadE),len(RadE)))
    Q=np.zeros(len(RadE))
    
    F0p[0]=-2*c/(3*0.5*(rho[0,k]+rho[0,k-1])*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1])*kappaT[0]+4.0)
    F0m[0]=-2*c/(3*0.5*(rho[0,k]+rho[0,k-1])*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1])*kappaT[0]+4.0)
    C[0,0]=m[0,k]/(dtk*rho[0,k])+0.5*m[0,k]*kappaA[0]*c*(1.0-nu[0])-0.5*(A[1,k-1]+A[1,k])*0.5*F0p[1]-0.5*(A[0,k-1]+A[0,k])*0.5*F0m[0]
    C[0,1]=0.5*F0p[1]*0.5*(A[1,k-1]+A[1,k])
    Q[0]=nu[0]*xi[0]+m[0,k]*kappaA[0]*c*(1.0-nu[0])*(a*Tp[0,k-1]**4-0.5*RadE[0,k-1])-(1./3.0)*RadEp[0,k-1]*(Ap[1,k-1]*0.5*(u[1,k-1]+u[1,k])-Ap[0,k-1]*0.5*(u[0,k-1]+u[0,k]))+ \
         0.5*(A[0,k-1]+A[0,k])*0.5*F0m[0]*(RadE[1,k-1]-a*TbL**4)-0.5*(A[1,k-1]+A[1,k])*0.5*F0p[1]*(RadE[1,k-1]-RadE[0,k-1])+m[0,k]*RadE[0,k-1]/(dtk*rho[0,k-1])

    F0p[-1]=-2*c/(3*0.5*(rho[-1,k]+rho[-1,k-1])*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1])*kappaT[-1]+4.0)
    F0m[-1]=-2*c/(3*0.5*(rho[-1,k]+rho[-1,k-1])*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1])*kappaT[-1]+4.0)
    C[-1,-1]=m[-1,k]/(dtk*rho[-1,k])+0.5*m[-1,k]*kappaA[-1]*c*(1.0-nu[-1])-0.5*(A[-2,k-1]+A[-2,k])*0.5*F0m[-2]-0.5*(A[-1,k-1]+A[-1,k])*0.5*F0p[-1]
    C[-1,-2]=0.5*F0m[-2]*0.5*(A[-2,k-1]+A[-2,k])
    Q[-1]=nu[-1]*xi[-1]+m[-1,k]*kappaA[-1]*c*(1.0-nu[-1])*(a*Tp[-1,k-1]**4-0.5*RadE[-1,k-1])-(1./3.0)*RadEp[-1,k-1]*(-Ap[-2,k-1]*0.5*(u[-2,k-1]+u[-2,k])+Ap[-1,k-1]*0.5*(u[-1,k-1]+u[-1,k]))+ \
         0.5*(A[-1,k-1]+A[-1,k])*0.5*F0p[-1]*(-a*TbR**4+RadE[-2,k-1])-0.5*(A[-2,k-1]+A[-2,k])*0.5*F0m[-2]*(RadE[-2,k-1]-RadE[-1,k-1])+m[-1,k]*RadE[-1,k-1]/(dtk*rho[-1,k-1])
            

#    C[0,0]=1.0;C[-1,-1]=1.0
#    Q[0]=RadE[0,k-1];Q[-1]=RadE[-1,k-1] #add real BCs
#    Q[0]=0.0;Q[-1]=0.0 #add real BCs
    for i in range(1,len(RadE)-1):
        C[i,i-1]=0.5*F0m[i]*0.5*(A[i,k-1]+A[i,k])
        C[i,i]=m[i,k]/(dtk*rho[i,k])+0.5*m[i,k]*kappaA[i]*c*(1.0-nu[i])-0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]
        C[i,i+1]=0.5*F0p[i]*0.5*(A[i+1,k-1]+A[i+1,k])
        Q[i]=nu[i]*xi[i]+m[i,k]*kappaA[i]*c*(1.0-nu[i])*(a*0.5*(Tp[i,k-1]**4+Tp[i,k]**4)-0.5*RadE[i,k-1])-(1./3.0)*0.5*(RadEp[i,k-1]+RadEp[i,k])*(Apk[i+1]*0.5*(u[i+1,k-1]+u[i+1,k])-Apk[i]*0.5*(u[i,k-1]+u[i,k]))+0.5*(A[i,k-1]+A[i,k])*0.5*F0m[i]*(RadE[i,k-1]-RadE[i-1,k-1])-0.5*(A[i+1,k-1]+A[i+1,k])*0.5*F0p[i]*(RadE[i+1,k-1]-RadE[i,k-1])+m[i,k]*RadE[i,k-1]/(dtk*rho[i,k-1])
    RadE[:,k]=np.linalg.solve(C,Q)
    return(RadE) 
    
def corrector_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho,Tp):
    a=0.01372
    c=299.792
    dtk=0.5*(dt_prev+dt)    
    kappaA=get_abs_opacity_corrector(T,k,TbL,TbR,Tp) 
    QP=artif_viscosity(P,gamma,u,r,rho,k)    
    Ppk=get_Ppk(P,k)
    Apk=get_Apk(A,k)    
    xi=np.zeros(len(P))
    for i in range(0,len(P)):
        xi[i]=-(Ppk[i]+QP[i])*(Apk[i+1]*0.5*(u[i+1,k-1]+u[i+1,k])- Apk[i]*0.5*(u[i,k-1]+u[i,k]))-(m[i,k]/dtk)*(e[i,k]-e[i,k-1])   
    for i in range(0,len(e)):
        e[i,k]=e[i,k-1]+(dtk*Cv[i]*(m[i,k]*c*kappaA[i]*(0.5*(RadE[i,k]+RadE[i,k-1])-a*0.5*(T[i,k-1]**4+T[i,k]**4))+xi[i]))/(m[i,k]*Cv[i]+dtk*m[i,k]*kappaA[i]*c*2*a*0.5*(T[i,k-1]**3+T[i,k]**3))
    return(e)     
    
    
def compute_energy_conservationLHS(u,m,rho,e,RadE,k):
    sum1=0.0;sum2=0.0;sum3=0.0;sum4=0.0;
    for i in range(0,len(u)-1):
        sum1+=0.5*m[i,k]*u[i,k]**2
    for i in range(0,len(e)):
        sum2+=m[i,k]*(e[i,k]+RadE[i,k]/rho[i,k])
    for i in range(0,len(u)-1):
        sum3+=0.5*m[i,k]*u[i,0]**2        
    for i in range(0,len(e)):
        sum4+=m[i,k]*(e[i,0]+RadE[i,0]/rho[i,0])
    tot=sum1+sum2-sum3-sum4
    ke=sum1-sum3
    ie=sum2-sum4    
    return(tot,ke,ie)

def compute_energy_conservationRHS(RHS,lin,lout,P,gamma,u,r,rho,k,RadE,A,dt,dt_prev,Tp,Pp,Ap,RadEp):
    QPk=artif_viscosity(P,gamma,u,r,rho,k)
    QPkm1=artif_viscosity(P,gamma,u,r,rho,k-1)
    kappaT=get_tot_opacity(T,k,TbL,TbR)
    a=0.01372
    c=299.792
    Ppk=get_Ppk(Pp,k)
    Apk=get_Apk(Ap,k)  
    RadEpk=get_RadEpk(RadEp,k)    
    dtk=0.5*(dt_prev+dt)        
    F0L=-2*c/(3*0.5*(rho[0,k]+rho[0,k-1])*(r[1,k]+r[1,k-1]-r[0,k]-r[0,k-1])*kappaT[0]+4.0)
    F0R=-2*c/(3*0.5*(rho[-1,k]+rho[-1,k-1])*(r[-1,k]+r[-1,k-1]-r[-2,k]-r[-2,k-1])*kappaT[-1]+4.0)
    RHS+=(0.5*(A[0,k]+A[0,k-1])*F0L-0.5*(A[-1,k]+A[-1,k-1])*F0R)*dtk+(Apk[0]*((1./3.)*RadEpk[0]+Ppk[0])*0.5*(u[0,k-1]+u[0,k]))*dtk-(Apk[-1]*((1./3.)*RadEpk[-1]+Ppk[-1])*0.5*(u[-1,k-1]+u[-1,k]))*dtk
    lin+=(0.5*(A[0,k]+A[0,k-1])*F0L)*dtk
    lout+=(0.5*(A[-1,k]+A[-1,k-1])*F0R)*dtk
    return(RHS,lin,lout)

    
    

    
'''         
a=0.01372
c=299.792
geometry='Slab'
N=10
Nt=3
gamma=1.5
Rmax=1.0
rho=np.ones((N,Nt))   
m=np.ones((N,Nt))   
Cv=np.ones(N)
u=np.ones((N+1,Nt))*0
u[:,0]=np.random.rand(N+1)*0
T=np.ones((N,Nt))
r,A,V=calc_grid_area_vol(N,Rmax,geometry,Nt)  
RadE=np.ones((N,Nt))*a
for i in range(0, N):
    for j in range(0,Nt):
        m[i,j]=rho[i,j]*V[i,j]
dt_vect=np.zeros(Nt)
e=np.zeros((N,Nt))
for i in range(0,N):e[i,:]=T[i,0]/Cv[i]    
P=np.ones((N,Nt))*5
for i in range(0,N):P[i,:]=(gamma-1)*rho[i,:]*e[i,:]
TbR=1.0
TbL=1.0
PbL=get_P(P,e,gamma,rho,1)[0][0]
PbR=get_P(P,e,gamma,rho,1)[-1][0]
RHS=0.0
for k in range(1,2):
    dt=get_dt(u,r,P,gamma,rho,k)
    dt_vect[k]=dt
    dt_prev=dt
    u=predictor_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma)
    print(u)
    r=get_coords(r,u,dt,dt_prev,k)
    A=get_area(A,k,r,geometry)
    V=get_vol(V,k,r,geometry)    
    rho=get_mass_dens(rho,m,V,k)
    RadE=predictor_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma)
    print(RadE)
    e=predictor_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho)
    print(e)
    T=get_T(T,e,Cv,k)
    P=get_P(P,e,gamma,rho,k)
    u=corrector_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma)
    print(u)
    r=get_coords(r,u,dt,dt_prev,k)
    A=get_area(A,k,r,geometry)
    V=get_vol(V,k,r,geometry)        
    RadE=corrector_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma)
    print(RadE)
    e=corrector_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho)
    print(e)
    lhs=compute_energy_conservationLHS(u,m,rho,e,RadE,k)
    RHS=compute_energy_conservationRHS(RHS,P,gamma,u,r,rho,k,RadE,A,dt,dt_prev)    
    
'''   
    
 #Morels test problem  
a=0.01372
c=299.792
geometry='Slab'
dx=0.02
Rmax=1.0
N=int(Rmax/dx)
Nt=4
gamma=5.0/3.0
rho=np.ones((N,Nt))   
m=np.ones((N,Nt))   
Cv=np.ones(N)*0.1
u=np.ones((N+1,Nt))*0
T=np.ones((N,Nt))*0.2
r,A,V=calc_grid_area_vol(N,Rmax,geometry,Nt)  
RadE=np.ones((N,Nt))*a*0.2**4
for i in range(0, N):
    for j in range(0,Nt):
        m[i,j]=rho[i,j]*V[i,j]
dt_vect=np.zeros(Nt)
e=np.zeros((N,Nt))
for i in range(0,N):e[i,:]=T[i,0]/Cv[i]    
P=np.ones((N,Nt))*5
for i in range(0,N):P[i,:]=(gamma-1)*rho[i,:]*e[i,:]
TbR=0.0
TbL=0.0
PbL=0.0#get_P(P,e,gamma,rho,1)[0][0]
PbR=0.0#get_P(P,e,gamma,rho,1)[-1][0]
RHS=0.0;lin=0.0;lout=0.0;
kev=np.zeros(Nt)
totv=np.zeros(Nt)
iev=np.zeros(Nt)
RHSv=np.zeros(Nt)
linv=np.zeros(Nt)
loutv=np.zeros(Nt) 
for k in range(1,Nt):
    dt=get_dt(u,r,P,gamma,rho,k)
    dt_vect[k]=dt
    dt_prev=dt
    u=predictor_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma)
    r=get_coords(r,u,dt,dt_prev,k)
    A=get_area(A,k,r,geometry)
    V=get_vol(V,k,r,geometry)    
    rho=get_mass_dens(rho,m,V,k)
    RadE=predictor_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma)
    print(RadE)
    e=predictor_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho)
 #   print(e)
    T=get_T(T,e,Cv,k)
    P=get_P(P,e,gamma,rho,k)
    Pp=P
    Tp=T
    RadEp=RadE
    Ap=A
    u=corrector_velocity(u,dt_prev,dt,m,A,r,rho,P,RadE,k,PbR,PbL,TbR,TbL,T,gamma,Tp)
    print(u)
    r=get_coords(r,u,dt,dt_prev,k)
    A=get_area(A,k,r,geometry)
    V=get_vol(V,k,r,geometry)     
    rho=get_mass_dens(rho,m,V,k)    
    RadE=corrector_rad_E(RadE,A,m,r,u,rho,T,P,Cv,TbL,TbR,dt_prev,dt,k,gamma,RadEp,Ap,Tp)
    print(RadE)
    e=corrector_internale(e,RadE,T,P,A,u,r,dt_prev,dt,k,Cv,m,gamma,rho,Tp)
    print(e)
    tot,ke,ie=compute_energy_conservationLHS(u,m,rho,e,RadE,k)
    kev[k]=ke
    totv[k]=tot
    iev[k]=ie
    RHS,lin,lout=compute_energy_conservationRHS(RHS,lin,lout,P,gamma,u,r,rho,k,RadE,A,dt,dt_prev,Tp,Pp,Ap,RadEp)    
    RHSv[k]=RHS
    linv[k]=lin
    loutv[k]=lout    