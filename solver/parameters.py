from firedrake import *
def defineSolverParameters():

    parameters_velo_direct={
            'ksp_type': 'lu',
            'ksp_rtol': 1.0e-7
    }
    parameters_pres_direct = { 'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'preonly',
                                        'pc_type': 'lu'}
    }
    parameters_corr_direct={
            "ksp_type": "cg",
            "ksp_rtol": 1e-8,
            'pc_type': 'ilu'
    }
##########################

    parameters_velo_iter_gamg={
        'ksp_type': 'fgmres',
        'ksp_monitor_true_residual': None,
        'ksp_rtol': 1e-6,
        'ksp_max_it': 500,
        'pc_type': 'gamg',
        'pc_gamg_sym_graph': True,
        'mg_levels': {'ksp_type': 'gmres',
                     'ksp_max_it': 5,
                     'pc_type': 'bjacobi',
                     'sub_pc_type': 'ilu'}
    }    

    parameters_velo_iter_ml={
        'ksp_type': 'fgmres',
        'ksp_rtol': 1e-6,
        'ksp_max_it': 500,
        'ksp_gmres_restart': 30,
        'pc_type': 'ml',
        'pc_mg_cycles': 1,
        'pc_ml_maxNlevels': 25,
        'ksp_monitor_true_residual': None,
        'mg_levels': {
                'ksp_type': 'richardson',
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu',
                'ksp_max_it': 5
        }        
    }   

    parameters_upd_iter_noprecon={
                        'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'cg',
                                            'pc_type': 'none',
                                            'ksp_rtol': 1e-6}
    }
    
    parameters_upd_iter_gamg={
                        'mat_type': 'matfree',
                        'ksp_type': 'preonly',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        'hybridization': {'ksp_type': 'cg',
                                            'pc_type': 'gamg',
                                            'ksp_rtol': 1e-6}
    }

    parameters_corr_iter={
            "ksp_type": "cg",
            "ksp_rtol": 1e-6,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
    }
########################

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
    spsc_iter_params=[parameters_velo_iter_gamg,parameters_upd_iter_gamg,parameters_corr_iter] 
    spsc_initial_params=[parameters_velo_initial]
    return [spsc_direct_params,spsc_iter_params,spsc_initial_params]
    #return [parameters_pres_iter,parameters_corr_iter,parameters_pres,parameters_pres_better,parameters_velo_iter,parameters_velo_initial,parameters_kovasznay]