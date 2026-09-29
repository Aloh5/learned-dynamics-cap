"""Discover and refine a KS candidate using fitted surrogates only.

No existing true-system periodic orbit is loaded. True RHS calls occur only in
training labels and frozen-candidate diagnostics. Every optimizer sees the
fitted polynomial model. Outputs are separate from the legacy Galerkin run.
"""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.io import savemat
from ks_neumann import (NeumannKSConfig, generate_attractor_trajectory,
    scan_ordinary_recurrences, resample_segment_to_phase_grid,
    refine_ordinary_orbit, refine_ordinary_orbit_matrix_free,
    time_fourier_coefficients, time_values_from_fourier, save_candidate,
    oversampled_residual, independent_flow_closure)
from surrogate_neumann import fit_surrogate, QuadraticSurrogate
from scan_block_proof import project_half_shift_symmetry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs_surrogate')
    parser.add_argument('--sample-time',type=float,default=800.)
    parser.add_argument('--candidate-rank',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    stages=[(16,16),(24,20),(40,28),(64,36),(80,40),(96,44),(128,52),(160,60)]
    models={}; metrics=[]
    for _,k in stages:
        path=out/f'surrogate_{k}.npz'; meta=out/f'surrogate_{k}.json'
        if args.resume and path.exists() and meta.exists():
            models[k]=QuadraticSurrogate.load(path); metric=json.loads(meta.read_text())
        else:
            models[k],metric=fit_surrogate(k); models[k].save(path)
            meta.write_text(json.dumps(metric,indent=2))
        metrics.append(metric); print('FIT',metric,flush=True)
    pd.DataFrame(metrics).to_csv(out/'surrogate_fit_metrics.csv',index=False)
    search=NeumannKSConfig(spatial_modes=16,temporal_modes=16,sample_time=args.sample_time,surrogate=models[16])
    path=out/'surrogate_trajectory.npz'
    if args.resume and path.exists():
        trajectory=np.load(path)['trajectory']
    else:
        _,trajectory=generate_attractor_trajectory(search)
        np.savez(path,trajectory=trajectory)
    returns=scan_ordinary_recurrences(trajectory,search.output_dt,min_period=20,max_period=120,max_candidates=30)
    returns.to_csv(out/'surrogate_recurrences.csv',index=False)
    candidate=returns.iloc[args.candidate_rank]
    print('RECURRENCE',candidate.to_dict(),flush=True)
    seed=resample_segment_to_phase_grid(trajectory[int(candidate.start_index):int(candidate.end_index)+1],16)
    period=float(candidate.period); coefficients=None; rows=[]
    for m,k in stages:
        config=NeumannKSConfig(spatial_modes=k,temporal_modes=m,full_grid_nodes=max(128,4*k),surrogate=models[k])
        path=out/f'stage_{m}_{k}.npz'; meta=out/f'stage_{m}_{k}.json'
        if args.resume and path.exists() and meta.exists():
            z=np.load(path); coefficients=z['coefficients']; period=float(z['period']); rows.append(json.loads(meta.read_text())); continue
        if coefficients is not None:
            old=(len(coefficients)-1)//2
            padded=np.zeros((2*m+1,k),complex)
            padded[m-old:m+old+1,:coefficients.shape[1]]=coefficients
            seed=time_values_from_fourier(padded)
        print('REFINE',m,k,period,flush=True)
        if m<=64:
            orbit=refine_ordinary_orbit(seed,period,config,max_nfev=260)
        else:
            orbit=refine_ordinary_orbit_matrix_free(seed,2*np.pi/period,config,max_nfev=60,tight=(m==160))
        if not orbit.optimizer_success or orbit.residual_linf>1e-7:
            raise RuntimeError(f'Surrogate refinement failed: {orbit}')
        period=orbit.period
        coefficients=time_fourier_coefficients(orbit.time_values)
        row=dict(temporal_modes=m,spatial_modes=k,period=period,surrogate_residual_rms=orbit.residual_l2,
            surrogate_residual_max=orbit.residual_linf,surrogate_optimizer_nfev=orbit.nfev,
            true_system_optimizer_calls=0,optimizer_message=orbit.message)
        print('STAGE',row,flush=True); rows.append(row)
        np.savez(path,coefficients=coefficients,period=period)
        meta.write_text(json.dumps(row,indent=2))
    # Projection is an explicit symmetry operation, never a true-RHS solve.
    projected=project_half_shift_symmetry(coefficients)
    projection_change=float(np.max(np.abs(projected-coefficients)))
    if projection_change>1e-9:
        raise RuntimeError(f'Orbit does not have required half-period symmetry: {projection_change}')
    coefficients=projected
    values=time_values_from_fourier(coefficients)
    from ks_neumann import collocation_residual
    surrogate_residual=collocation_residual(values,2*np.pi/period,config)
    if np.max(np.abs(surrogate_residual))>1e-7:
        raise RuntimeError('Projected surrogate candidate failed residual check')
    data=dict(frequency=2*np.pi/period,period=period,velocity_time_fourier=coefficients,
        velocity_time_values=values,reference_time_values=values,interval_length=22.,
        extension_length=44.,spatial_modes=60,temporal_modes=160,newton_corrections=0,
        true_system_optimizer_calls=0,candidate_origin='data_fitted_surrogate_only',
        symmetry_projection_max_change=projection_change)
    for filename in ('ks_neumann_full_pde_certificate.json', 'ks_neumann_cap_summary.csv'):
        (out/filename).unlink(missing_ok=True)
    savemat(out/'ks_neumann_cap_candidate.mat',data)
    np.savez(out/'ks_neumann_cap_candidate.npz',**data)
    pd.DataFrame(rows).to_csv(out/'surrogate_continuation_stages.csv',index=False)
    # Freeze before any true-system validation calls.
    true_config=replace(config,surrogate=None)
    _,residual=oversampled_residual(values,2*np.pi/period,true_config)
    diagnostics=independent_flow_closure(values,period,true_config)
    diagnostics.update(period=period,true_oversampled_residual_rms=float(np.sqrt(np.mean(residual**2))),
        true_oversampled_residual_max=float(np.max(np.abs(residual))),
        surrogate_retained_residual_rms=float(np.sqrt(np.mean(surrogate_residual**2))),
        true_system_optimizer_calls=0,newton_corrections=0,
        symmetry_projection_max_change=projection_change,
        candidate_origin='data_fitted_surrogate_only',proof_status='not_evaluated_by_candidate_generator')
    (out/'candidate_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print('FROZEN',diagnostics,flush=True)

if __name__=='__main__':
    main()
