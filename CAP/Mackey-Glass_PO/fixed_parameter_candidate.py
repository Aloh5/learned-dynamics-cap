"""Finite candidate construction; numerical calculations here are not the proof."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]/'notebooks/mackey_glass/outputs_zero_correction/MG_zero_correction_candidate.mat'
OUT = HERE/'outputs_fixed_parameter'

def coefficients(values, order):
    c=np.fft.fft(values)/len(values)
    return np.r_[c[-order:],c[:order+1]]

def evaluate(u, size=8192):
    n=(len(u)-1)//2
    c=np.zeros(size,complex);c[:n+1]=u[n:];c[-n:]=u[:n]
    return np.fft.ifft(c).real*size

def nonlinear(u, order):
    v=evaluate(u)
    return coefficients(v/(1+v**10),order), coefficients((1-9*v**10)/(1+v**10)**2,order)

def system(omega,u,reference,phase):
    n=(len(u)-1)//2;k=np.arange(-n,n+1);nz=k!=0
    g,gp=nonlinear(u,2*n)
    gs=g[n:3*n+1];shift=np.exp(-1j*k*omega*1.7)
    j=np.ones(len(k),complex);j[nz]=1/(1j*k[nz]*reference)
    f=u+j*(u-2*shift*gs);f[n]=u[n]-2*gs[n]
    f[nz]+=(omega/reference-1)*u[nz]
    mult=gp[(k[:,None]-k[None,:])+2*n]
    du=np.diag(np.where(nz,omega/reference,1))+j[:,None]*(np.eye(len(k))-2*shift[:,None]*mult)
    du[n,:]=-2*mult[n,:];du[n,n]+=1
    dw=(u+2*1.7*shift*gs)/reference;dw[n]=0
    jac=np.block([[np.zeros((1,1)),phase[None,:]],[dw[:,None],du]])
    return f,jac

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--newton-steps',type=int,default=0);ap.add_argument('--order',type=int,default=80)
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    raw=loadmat(SOURCE);a=raw['candidate_real_fourier'].ravel();src_n=(len(a)-1)//2
    positive=(a[1::2]-1j*a[2::2])/2
    src=np.r_[positive.conj()[::-1],a[0],positive]
    n=args.order;u=src[src_n-n:src_n+n+1].copy();initial=u.copy()
    omega=float(2*np.pi/(.1*raw['period_candidate_physical'].item()));reference=omega
    k=np.arange(-n,n+1);tangent=1j*k*u
    phase=tangent.conj()/np.vdot(tangent,tangent).real
    phase=(phase+phase[::-1].conj())/2
    target=(phase@u).real
    history=[]
    for step in range(args.newton_steps+1):
        f,jac=system(omega,u,reference,phase)
        full=np.r_[(phase@u).real-target,f]
        history.append({'step':step,'residual_l1':float(np.sum(np.abs(full))), 'omega':omega, 'physical_period':2*np.pi/(.1*omega)})
        if step==args.newton_steps:break
        correction=np.linalg.solve(jac,full)
        omega-=correction[0].real;u-=correction[1:]
        u=(u+u[::-1].conj())/2
    # Approximate inverse is exact dyadic data in MATLAB, validated there.
    inv=np.linalg.inv(jac);perm=np.r_[0,np.arange(len(u),0,-1)]
    inv=(inv+inv[np.ix_(perm,perm)].conj())/2
    q=160;v=evaluate(u);recip=coefficients(1/(1+v**10),q)
    recip=(recip+recip[::-1].conj())/2
    poles=np.exp(1j*np.pi*(2*np.arange(10)+1)/10)
    pole_reciprocals=np.stack([coefficients(1/(v-z),q) for z in poles],axis=1)
    (OUT/'mg_fixed_parameter_certificate.json').unlink(missing_ok=True)
    savemat(OUT/'mg_fixed_parameter_candidate.mat',dict(u=u[:,None],omega=omega,reference_omega=reference,phase=phase[None,:],phase_target=target,A=inv,reciprocal=recip[:,None],pole_reciprocals=pole_reciprocals,newton_steps=args.newton_steps,initial_u=initial[:,None],initial_omega=reference))
    record={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'true_system_newton_steps':args.newton_steps,'order':n,'source_order':src_n,'history':history,'source_truncation_weighted_l1':float(sum(abs(src[:src_n-n])*1.01**abs(np.arange(-src_n,-n)))+sum(abs(src[src_n+n+1:])*1.01**np.arange(n+1,src_n+1))),'frequency_change':float(abs(omega-reference)),'state_change_weighted_l1':float(np.sum(abs(u-initial)*1.01**abs(k)))}
    (OUT/'candidate_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
