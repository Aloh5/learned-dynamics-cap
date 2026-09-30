import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cap-matplotlib')
import ast,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
root=Path(__file__).resolve().parents[1]; out=root/'results/figures'
out.mkdir(parents=True, exist_ok=True)
def funcs(path,names):
 nb=json.loads((root/path).read_text()); source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'); tree=ast.parse(source); tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 ns={'np':np};exec(compile(tree,str(path),'exec'),ns);return ns
ks=funcs('notebooks/kuramoto_sivashinsky/Kuramoto.ipynb',{'rfft_wavenumbers','dealiased_nonlinear_rfft','simulate_ks_etdrk4','ks_rhs_spectral_state'})
def field(data):
 p=data['powers']; coeff=data['coefficients']; dim=p.shape[1]
 indices=np.full((len(p),int(p.sum(1).max())),dim,dtype=int)
 for i,row in enumerate(p):
  ind=np.repeat(np.arange(dim),row);indices[i,:len(ind)]=ind
 def fun(x): return coeff@np.prod(np.r_[x,1.0][indices],axis=1)
 return fun

def draw(a,b,name,title,labels):
 fig=plt.figure(figsize=(11,5.2))
 fig.subplots_adjust(left=0.01,right=0.91,bottom=0.14,top=0.84,wspace=0.12)
 axes=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
 both=np.r_[a,b]; lo=both.min(0);hi=both.max(0);pad=.05*(hi-lo)
 for ax,z,c,t in zip(axes,[a,b],['#b63a43','#216894'],['Reference','Learned surrogate']):
  ax.plot(*z.T,color=c,lw=.35,alpha=.72,rasterized=True)
  ax.set(title=t,xlabel=labels[0],ylabel=labels[1],zlabel=labels[2],xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),zlim=(lo[2]-pad[2],hi[2]+pad[2]))
  ax.view_init(elev=23,azim=-55);ax.tick_params(labelsize=8);ax.set_box_aspect((1,1,1))
 fig.suptitle(title,fontsize=14)
 fig.savefig(out/name,dpi=240);plt.close(fig)

mg=np.load(root/'results/equilibria/mackey_glass_model.npz'); f=field(mg)
# Verify stored coefficients match the manuscript's normalized degree-weighted fit.
x=mg['training'];p=mg['powers'];theta=np.prod(x[:,None,:]**p[None,:,:],2);sc=np.sqrt(np.mean(theta**2,0));sc[sc<1e-14]=1;tn=theta/sc;w=np.maximum(p.sum(1),1);w[0]=0
refit=np.linalg.solve(tn.T@tn+1e-4*np.diag(w),tn.T@np.gradient(x,.1,axis=0,edge_order=2)).T/sc
print('MG relative coefficient check',np.linalg.norm(refit-mg['coefficients'])/np.linalg.norm(mg['coefficients']),flush=True)
assert np.allclose(refit,mg['coefficients'],rtol=1e-6,atol=1e-7)
v=np.full(50000+171,1.2)
for i in range(170,len(v)-1): v[i+1]=v[i]+.1*(.2*v[i-170]/(1+v[i-170]**10)-.1*v[i])
s=v[170:]; reference=np.column_stack([s[35000-j*45:45000-j*45] for j in range(3)])
initial=np.array([s[15000-j*45] for j in range(6)]);trajectory=np.empty((20001,6));trajectory[0]=initial
for i in range(20000):
 x=trajectory[i];a=f(x);b=f(x+.05*a);c=f(x+.05*b);d=f(x+.1*c);trajectory[i+1]=x+.1*(a+2*b+2*c+d)/6
assert np.isfinite(trajectory).all() and trajectory.min()>.15 and trajectory.max()<1.65
learned=trajectory[5000:,:3]
draw(reference,learned,'mg_attractor_comparison.png','Mackey–Glass: sampled long-time geometry',['$X_1$','$X_2$','$X_3$'])
print('MG figure complete',flush=True)
model=np.load(root/'results/equilibria/periodic_ks_model.npz');f=field(model);initial=model['training'][-1]
reference=ks['simulate_ks_etdrk4'](initial,22.,.025,40000)[::4]
times=np.arange(10001)*.1
sol=solve_ivp(lambda t,x:f(x),(0,1000),initial,t_eval=times,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.1)
assert sol.success and sol.y.shape==(14,10001) and np.isfinite(sol.y).all()
a=reference[2000:];b=sol.y.T[2000:];mu=a.mean(0);_,ss,vh=np.linalg.svd(a-mu,full_matrices=False);basis=vh[:3].T
pc_a=(a-mu)@basis;pc_b=(b-mu)@basis
variance=float((ss[:3]**2).sum()/(ss**2).sum())
draw(pc_a,pc_b,'ks_periodic_attractor_comparison.png',f'Periodic KS: shared PCA coordinates ({variance:.1%} reference variance)',['PC1','PC2','PC3'])
np.savez_compressed(root/'results/figures/trajectories.npz',mg_reference=reference if False else np.column_stack([s[35000-j*45:45000-j*45] for j in range(3)]),mg_learned=learned,ks_reference=a,ks_learned=b,ks_pca_basis=basis,ks_pca_mean=mu)
(out/'metrics.json').write_text(json.dumps({'KS_reference_variance':variance,'KS_reference_max_abs':float(abs(a).max()),'KS_surrogate_max_abs':float(abs(b).max()),'MG_coefficient_check':float(np.linalg.norm(refit-mg['coefficients'])/np.linalg.norm(mg['coefficients']))},indent=2))
print('KS figure complete; variance',variance,flush=True)
