from firedrake import *
def defineSolverParameters():

    parameters_velo_direct={
            'ksp_type': 'lu',
            'ksp_rtol': 1.0e-7
    }
    parameters_velo_iter={
            'pc_type': 'gamg',
            'ksp_type': 'gmres',
            'ksp_rtol': 1.0e-8
    }


    parameters_pres_direct = { 'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'preonly',
                                        'pc_type': 'lu'}
    }

    parameters_pres_iter={
                        'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'gamg',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'cg',
                                            'pc_type': 'none',
                                            'ksp_rtol': 1e-8
                        }
    }

    parameters_pres_better={
                        'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'cg',
                                            'pc_type': 'none',
                                            'ksp_rtol': 1e-8,
                                            'mat_type': 'matfree'}
    }

    parameters_velo_initial={
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_monitor_true_residual": None,
    "ksp_view": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC", 
    "fieldsplit_1_Mp_ksp_type": "preonly",
    "fieldsplit_1_Mp_pc_type": "ilu"
    }

    parameters_corr_direct={
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            'pc_type': 'ilu'
    }

    parameters_corr_iter={
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            'pc_type': 'ilu'
    }

    parameters_kovasznay={  
            "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
         "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_0_ksp_rtol": 1e-8,
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_ksp_rtol": 1e-8,
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
     #     "ksp_type": "gmres",
     #   "ksp_converged_reason": None,
     #   "ksp_gmres_restart":100,
     #   "ksp_rtol":1e-12,
     #   "pc_type":"lu",
     #   "pc_factor_mat_solver_type": "mumps",
     #   "mat_type":"aij"
    }

    spsc_direct_params=[parameters_velo_direct,parameters_pres_direct,parameters_corr_direct] 
    spsc_iter_params=[parameters_velo_iter,parameters_pres_better,parameters_corr_iter] 
    spsc_initial_params=[parameters_velo_initial]
    return [spsc_direct_params,spsc_iter_params,spsc_initial_params]
    #return [parameters_pres_iter,parameters_corr_iter,parameters_pres,parameters_pres_better,parameters_velo_iter,parameters_velo_initial,parameters_kovasznay]