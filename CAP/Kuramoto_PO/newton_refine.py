"""Newton--GMRES refinement of the original Neumann KS Fourier equation.

Uses the original certified preconditioner and phase slice. Floating-point
results from this module are candidates, not proofs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import LinearOperator, gmres
from diagnose_smooth_cap import real_time_coefficients, temporal_derivative
from scan_block_proof import core_layout, project_half_shift_symmetry

HERE=Path(__file__).resolve().parent
BASE=HERE/'outputs_surrogate'


def complex_time(real):
    m=(len(real)-1)//2
    positive=(real[1:m+1]-1j*real[m+1:])/2
    return np.vstack((positive[::-1].conj(),real[0:1],positive))


def derivative(real):
    m=(len(real)-1)//2
    out=np.zeros_like(real)
    out[1:m+1]=np.arange(1,m+1)[:,None]*real[m+1:]
    out[m+1:]=-np.arange(1,m+1)[:,None]*real[1:m+1]
    return out


def product_derivative(a,b):
    """Exact-support numerical Fourier evaluation of d_y(a*b) in sine basis."""
    ma=(len(a)-1)//2;mb=(len(b)-1)//2
    ka=a.shape[1];kb=b.shape[1]
    nt=1<<int(2*(ma+mb)+1).bit_length()
    nx=1<<int(2*(ka+kb)+1).bit_length()
    def values(u):
        m=(len(u)-1)//2;k=u.shape[1]
        c=complex_time(u)
        grid=np.zeros((nt,nx),complex)
        rows=np.arange(-m,m+1)%nt
        grid[np.ix_(rows,np.arange(1,k+1))]=c/(2j)
        grid[np.ix_(rows,(-np.arange(1,k+1))%nx)]=-c/(2j)
        return np.fft.ifft2(grid).real*(nt*nx)
    coeff=np.fft.fft2(values(a)*values(b))/(nt*nx)
    # d_y converts even cosine modes into odd sine modes: a_k=-2*k*c_k.
    m=ma+mb;k=ka+kb
    centered=-2*np.arange(1,k+1)[None,:]*coeff[np.ix_(np.arange(-m,m+1)%nt,np.arange(1,k+1))]
    return real_time_coefficients(centered,m)


def inverse_symbol(real,f,nu,shifted=False):
    m=(len(real)-1)//2;k=np.arange(1,real.shape[1]+1)
    p=nu*k**4-k**2+(1/nu if shifted else 0)
    out=np.empty_like(real);out[0]=real[0]/p
    z=f*np.arange(1,m+1)[:,None];den=p*p+z*z
    out[1:m+1]=(p*real[1:m+1]-z*real[m+1:])/den
    out[m+1:]=(z*real[1:m+1]+p*real[m+1:])/den
    return out


def pad_real(u,m,k):
    old=(len(u)-1)//2
    out=np.zeros((2*m+1,k));out[0,:u.shape[1]]=u[0]
    out[1:old+1,:u.shape[1]]=u[1:old+1]
    out[m+1:m+old+1,:u.shape[1]]=u[old+1:]
    return out


class Refiner:
    def __init__(self,m=220,k=100):
        self.m=m;self.k=k;self.nu=(np.pi/22)**2
        self.source=loadmat(BASE/'ks_neumann_cap_candidate.mat')
        proof=loadmat(BASE/'ks_neumann_fourier_proof_data.mat',variable_names=['finite_inverse','source_frequency','projected_velocity_time_fourier'])
        self.B=proof['finite_inverse']
        self.original_physical=project_half_shift_symmetry(self.source['velocity_time_fourier'])
        assert np.array_equal(proof['projected_velocity_time_fourier'],self.original_physical)
        assert proof['source_frequency'].item()==self.source['frequency'].item()
        self.f0=self.source['frequency'].item()/self.nu
        self.original=real_time_coefficients(self.original_physical/(np.pi/22),160)
        self.u0=pad_real(self.original,m,k)
        _,self.allowed=core_layout(m,k)
        _,oldallowed=core_layout(160,40)
        self.coreflat=np.array([((i//40 if i//40<=160 else m+(i//40-160))*k+i%40) for i in oldallowed])
        self.corepositions=np.searchsorted(self.allowed,self.coreflat)
        assert np.array_equal(self.allowed[self.corepositions],self.coreflat)
        self.tangent0=derivative(self.u0).ravel()[self.coreflat]
        self.dimension=1+len(self.allowed)

    def unpack(self,z):
        u=np.zeros((2*self.m+1,self.k));u.ravel()[self.allowed]=z[1:]
        return z[0],u

    def pack(self,f,u):return np.r_[f,u.ravel()[self.allowed]]

    def precondition(self,phase,e):
        tail=inverse_symbol(e,self.f0,self.nu)
        core=inverse_symbol(e,self.f0,self.nu,True).ravel()[self.coreflat]
        result=np.r_[0.,tail.ravel()[self.allowed]]
        corrected=self.B@np.r_[phase,core]
        result[0]=corrected[0];result[1+self.corepositions]=corrected[1:]
        return result

    def equation(self,f,u):
        product=product_derivative(u,u)/2
        linear=f*derivative(u)+(self.nu*np.arange(1,self.k+1)**4-np.arange(1,self.k+1)**2)*u
        # Restrict only after computing the full de-aliased quadratic.
        rows=np.r_[0,np.arange(1,self.m+1),2*self.m+np.arange(1,self.m+1)]
        ecore=product[np.ix_(rows,np.arange(self.k))]+linear
        return ecore

    def residual(self,z):
        f,u=self.unpack(z)
        phase=np.dot(self.tangent0,(u-self.u0).ravel()[self.coreflat])
        return self.precondition(phase,self.equation(f,u))

    def jacobian_action(self,z,h):
        f,u=self.unpack(z);df,v=self.unpack(h)
        prod=product_derivative(u,v)
        rows=np.r_[0,np.arange(1,self.m+1),2*self.m+np.arange(1,self.m+1)]
        e=f*derivative(v)+df*derivative(u)+(self.nu*np.arange(1,self.k+1)**4-np.arange(1,self.k+1)**2)*v+prod[np.ix_(rows,np.arange(self.k))]
        return self.precondition(np.dot(self.tangent0,v.ravel()[self.coreflat]),e)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--temporal-modes',type=int,default=220);ap.add_argument('--spatial-modes',type=int,default=100);ap.add_argument('--steps',type=int,default=3)
    ap.add_argument('--output-dir',type=Path,default=HERE/'outputs_newton_refined')
    args=ap.parse_args();out=args.output_dir
    if out.resolve()==BASE.resolve():ap.error('Refined outputs must be separate from original proof inputs.')
    if args.temporal_modes<160 or args.spatial_modes<60:ap.error('Must retain all original modes.')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('ks_newton_refined_certificate.json','newton_refinement_manifest.json'):(out/name).unlink(missing_ok=True)
    r=Refiner(args.temporal_modes,args.spatial_modes)
    z=r.pack(r.f0,r.u0);history=[]
    for step in range(args.steps+1):
        residual=r.residual(z)
        row={'step':step,'preconditioned_core_residual_l1':float(np.linalg.norm(residual,1)),'scaled_frequency':z[0],'physical_period':2*np.pi/(r.nu*z[0])}
        history.append(row);print(json.dumps(row),flush=True)
        if step==args.steps:break
        operator=LinearOperator((r.dimension,r.dimension),matvec=lambda h:r.jacobian_action(z,h))
        delta,info=gmres(operator,residual,atol=0.,rtol=1e-7,restart=60,maxiter=10)
        if info:raise RuntimeError(f'Newton linear solve failed: {info}')
        row['newton_correction_l1']=float(np.linalg.norm(delta,1));z-=delta
    f,u=r.unpack(z)
    displacement=abs(f-r.f0)+np.sum(abs(u-r.u0))
    record={'true_system_newton_steps':args.steps,'history':history,'temporal_modes':r.m,'spatial_modes':r.k,'scaled_center_displacement_l1':float(displacement),'phase_defect':float(np.dot(r.tangent0,(u-r.u0).ravel()[r.coreflat])),'source_sha256':hashlib.sha256((BASE/'ks_neumann_cap_candidate.mat').read_bytes()).hexdigest(),'preconditioner_source':'Original certified 160-time/40-space core inverse and original-frequency tail symbols; no scan reused as a new-center scan.'}
    savemat(out/'ks_newton_refined_candidate.mat',{'scaled_real_coefficients':u,'scaled_frequency':f,'newton_steps':args.steps,'original_scaled_frequency':r.f0,'interval_length':22.,'temporal_modes':r.m,'spatial_modes':r.k})
    (out/'candidate_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
