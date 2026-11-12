% Refined Neumann KS proof using the original certified fixed operator.
% No new finite/tail scan is asserted: the original full-space derivative
% bound is transferred analytically using the global quadratic remainder.
clear
script_dir=fileparts(mfilename('fullpath'));
base=fullfile(script_dir,'outputs_surrogate');
out=fullfile(script_dir,'outputs_newton_refined');
if ~isempty(getenv('KS_REFINED_OUTPUT_DIR')),out=getenv('KS_REFINED_OUTPUT_DIR');end
assert(~strcmp(out,base),'Preserve the uncorrected certificate.');
certfile=fullfile(out,'ks_newton_refined_certificate.json');
if exist(certfile,'file'),delete(certfile);end
D=load(fullfile(out,'ks_newton_refined_candidate.mat'));
original=load(fullfile(base,'ks_neumann_cap_candidate.mat'));
P=load(fullfile(base,'ks_neumann_fourier_proof_data.mat'));
old=jsondecode(fileread(fullfile(base,'ks_neumann_full_pde_certificate.json')));
assert(old.pde_proof_success && old.tail_bound_complete && old.newton_corrections==0);
assert(D.newton_steps>0 && D.interval_length==22 && original.interval_length==22);
assert(original.frequency==P.source_frequency);
assert(old.frequency==original.frequency && old.period==original.period);
assert(original.newton_corrections==0 && original.true_system_optimizer_calls==0);
physical=original.velocity_time_fourier;
assert(isequal(physical,P.projected_velocity_time_fourier));
assert(P.temporal_modes==160 && P.spatial_modes==40 && P.dimension==6421);
assert(old.temporal_modes==160 && old.spatial_modes==40 && old.finite_dimension==6421);
q=intval('pi')/22;nu=q^2;f0=intval(original.frequency)/nu;
f=intval(D.scaled_frequency);U=intval(D.scaled_real_coefficients);
M=(size(U,1)-1)/2;K=size(U,2);
assert(M==D.temporal_modes && K==D.spatial_modes && M>=160 && K>=60);
U0=intval(zeros(321,60));U0(1,:)=intval(real(physical(161,:)))/q;
U0(2:161,:)=2*intval(real(physical(162:end,:)))/q;
U0(162:end,:)=-2*intval(imag(physical(162:end,:)))/q;
original_padded=pad_real(U0,M,K);
for j=1:2*M+1
    if j<=M+1,n=j-1;else,n=j-M-1;end
    assert(all(sup(abs(U(j,mod(n+(1:K),2)~=0)))==0),'Wrong symmetry.');
end
center_distance=abs(f-f0)+sum(abs(U-original_padded),'all');
alpha=intval(old.alpha)+2*intval(old.Z2)*center_distance;
Z2=intval(old.Z2);
assert(sup(alpha)<1,'Transferred full-space derivative bound is not contractive.');
% Original finite tangent and target define exactly the same phase slice.
tangent0=derivative(U0);
phase=core_vector(tangent0,160,40)'* ...
    (core_vector(U,160,40)-core_vector(U0,160,40));

% Direct outward-rounded coefficient convolution, with no FFT or quadrature.
% This replaces trigonometric quadrature only for the residual evaluation;
% all original linear-defect and bilinear estimates remain in force.
C=intval(zeros(2*M+1,K));C(M+1,:)=U(1,:);
C(M+2:end,:)=(U(2:M+1,:)-1i*U(M+2:end,:))/2;
C(1:M,:)=conj(flipud(C(M+2:end,:)));
Tout=2*M;Kout=2*K;
nonlinear=intval(zeros(2*Tout+1,Kout));
for j=1:K
    aidx=find(sup(abs(C(:,j)))~=0);a=C(aidx,j);
    if isempty(aidx),continue;end
    assert(all(diff(aidx)==2),'Expected temporal parity lattice.');
    for k=j:K
        bidx=find(sup(abs(C(:,k)))~=0);b=C(bidx,k);
        if isempty(bidx),continue;end
        assert(all(diff(bidx)==2),'Expected temporal parity lattice.');
        len=length(a)+length(b)-1;
        index=(0:len-1)'-(0:length(b)-1)+1;
        valid=index>=1 & index<=length(a);
        convolution=intval(zeros(size(index)));
        convolution(valid)=a(index(valid));
        convolution=convolution*b;
        output_indices=aidx(1)+bidx(1)-1+2*(0:len-1);
        multiplicity=1+(j~=k);
        nonlinear(output_indices,j+k)=nonlinear(output_indices,j+k)+...
            intval(multiplicity*(j+k)/4)*convolution;
        if j~=k
            nonlinear(output_indices,k-j)=nonlinear(output_indices,k-j)-...
                intval(multiplicity*(k-j)/4)*convolution;
        end
    end
    if mod(j,10)==0,fprintf('Verified coefficient convolutions: spatial input %d/%d\n',j,K);end
end
R=intval(zeros(2*Tout+1,Kout));R(1,:)=real(nonlinear(Tout+1,:));
R(2:Tout+1,:)=2*real(nonlinear(Tout+2:end,:));
R(Tout+2:end,:)=-2*imag(nonlinear(Tout+2:end,:));
linear=f*derivative(U)+(nu*intval(1:K).^4-intval(1:K).^2).*U;
R=R+pad_real(linear,Tout,Kout);
% Analytically invariant parity removes coefficients known exactly to vanish.
for j=1:2*Tout+1
    if j<=Tout+1,n=j-1;else,n=j-Tout-1;end
    R(j,mod(n+(1:Kout),2)~=0)=0;
end
smooth=inverse_symbol(R,f0,nu,true);
raw=inverse_symbol(R,f0,nu,false);
core=core_vector(smooth,160,40);
finite=intval(P.finite_inverse)*[phase;core];
mask=core_mask(size(raw),160,40);
Y=sum(abs(finite))+sum(abs(raw(~mask)));
radtext=getenv('KS_REFINED_RADIUS');
if isempty(radtext),radtext='7e-9';end
r=intval(radtext);
p=Y-(1-alpha)*r+Z2*r^2;
contraction=alpha+2*Z2*r;
fprintf('Refined KS: Y <= %.16e, alpha <= %.16e, Z2 <= %.16e, displacement <= %.16e\n',...
    sup(Y),sup(alpha),sup(Z2),sup(center_distance));
fprintf('Radius %s, p <= %.16e, contraction <= %.16e\n',radtext,sup(p),sup(contraction));
assert(inf(r)>0 && sup(p)<0 && sup(contraction)<1,'Refined radii inequalities failed.');
% Containment identifies the same original solution, without changing phase.
contained=center_distance+r;
assert(sup(contained)<old.validation_radius,'Refined ball outside original certificate.');
nonconstant=max(inf(abs(U(2:end,:))),[],'all')-sup(r);
assert(nonconstant>0 && inf(f-r)>0);
period=2*intval('pi')/(nu*(f+infsup(-sup(r),sup(r))));
certificate=struct('pde_proof_success',true,'tail_bound_complete',true,...
    'same_orbit_as_original',true,'phase_slice_identical',true,...
    'formulation','Neumann potential / odd velocity','newton_corrections',D.newton_steps,...
    'temporal_modes',M,'spatial_modes',K,'inverse_temporal_modes',160,'inverse_spatial_modes',40,...
    'norm','abs(delta f) + sum abs(real trigonometric delta v amplitudes)',...
    'Y',sup(Y),'alpha',sup(alpha),'Z2',sup(Z2),...
    'validation_radius',sup(r),'radius_decimal',radtext,...
    'radius_polynomial',sup(p),'contraction_bound',sup(contraction),...
    'center_distance_upper',sup(center_distance),'containment_upper',sup(contained),...
    'original_radius',old.validation_radius,...
    'true_period_lower',inf(period),'true_period_upper',sup(period),...
    'time_dependent_coefficient_lower',nonconstant,...
    'phase_residual_upper',sup(abs(phase)),...
    'residual_temporal_output_modes',Tout,'residual_spatial_output_modes',Kout,...
    'residual_evaluation','direct outward-rounded temporal coefficient convolution and exact sine product identities',...
    'fixed_preconditioner','original certified finite inverse and original-frequency tail',...
    'derivative_transfer','alpha_new <= alpha_original + 2 Z2 distance; Z2 unchanged (quadratic equation)',...
    'interval_arithmetic','INTLAB 13');
fid=fopen(certfile,'w');fprintf(fid,'%s\n',jsonencode(certificate,PrettyPrint=true));fclose(fid);
save(fullfile(out,'ks_newton_refined_bounds.mat'),'Y','alpha','Z2','r','p','contraction','center_distance','contained','period','phase');
disp(certificate)

function d=derivative(U)
M=(size(U,1)-1)/2;d=intval(zeros(size(U)));
d(2:M+1,:)=intval((1:M)').*U(M+2:end,:);
d(M+2:end,:)=-intval((1:M)').*U(2:M+1,:);
end
function V=pad_real(U,M,K)
m=(size(U,1)-1)/2;k=size(U,2);V=intval(zeros(2*M+1,K));
V(1,1:k)=U(1,:);V(2:m+1,1:k)=U(2:m+1,:);V(M+2:M+m+1,1:k)=U(m+2:end,:);
end
function vec=core_vector(U,M,K)
m=(size(U,1)-1)/2;rows=[1,2:M+1,m+2:m+M+1];
block=U(rows,1:K).';block=block(:);
harmonics=[0,1:M,1:M];allow=mod(repelem(harmonics,K)'+repmat((1:K)',2*M+1,1),2)==0;
vec=block(allow);
end
function mask=core_mask(sz,M,K)
m=(sz(1)-1)/2;mask=false(sz);
for n=0:M
    ks=find(mod(n+(1:K),2)==0);mask(n+1,ks)=true;
    if n>0,mask(m+n+1,ks)=true;end
end
end
function result=inverse_symbol(U,f,nu,shifted)
m=(size(U,1)-1)/2;k=intval(1:size(U,2));p=nu*k.^4-k.^2;
if shifted,p=p+1/nu;end
result=intval(zeros(size(U)));result(1,:)=U(1,:)./p;
z=f*intval((1:m)');den=p.^2+z.^2;
result(2:m+1,:)=(p.*U(2:m+1,:)-z.*U(m+2:end,:))./den;
result(m+2:end,:)=(z.*U(2:m+1,:)+p.*U(m+2:end,:))./den;
end
