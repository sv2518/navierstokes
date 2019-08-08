

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def solveassembly_internal(columns,axis10,axis11,order):
    #gather internal data
    timeassemblypred_int=columns["jac_eval_time_pred"]
    timesolvepred_int=columns["snes_time_pred"]
    timeassemblyupd_int=columns["jac_eval_time_upd"]
    timesolveupd_int=columns["snes_time_upd"]
    timeassemblycorr_int=columns["jac_eval_time_corr"]
    timesolvecorr_int=columns["snes_time_corr"]

    #collect times
    timeassembly_int=timeassemblypred_int+timeassemblyupd_int+timeassemblycorr_int
    timesolve_int=timesolvepred_int+timesolveupd_int+timesolvecorr_int
    timepred_int=timeassemblypred_int+timesolvepred_int
    timeupd_int=timeassemblyupd_int+timesolveupd_int
    timecorr_int=timeassemblycorr_int+timesolvecorr_int

    #relative plot for parts
    fig9= plt.figure(9)
    axis9 = fig9.gca()
    a12=axis9.bar(1, timeassemblypred_int/timepred_int, width,
                 linewidth=1,color="red")
    b=timeassemblypred_int/timepred_int
    a13=axis9.bar(1, timesolvepred_int/timepred_int, width,
                 linewidth=1,color="blue",bottom=b)

    a14=axis9.bar(2, timeassemblyupd_int/timeupd_int, width,
                 linewidth=1,color="red")
    b=timeassemblyupd_int/timeupd_int
    a15=axis9.bar(2, timesolveupd_int/timeupd_int, width,
                 linewidth=1,color="blue",bottom=b)

    a16=axis9.bar(3, timeassemblycorr_int/timecorr_int, width,
                 linewidth=1,color="red")
    b=timeassemblycorr_int/timecorr_int
    a17=axis9.bar(3, timesolvecorr_int/timecorr_int, width,
                 linewidth=1,color="blue",bottom=b)
    
    axis9.legend((a12[0],a13[0]),
    ("assembly (snes jacobian eval)", "solve (KSP solve)"))
    axis9.set_xlabel('Parts')
    axis9.set_ylabel('Normalised Time')
    fig9.savefig("solveassembly_internal/tasparts_relative_internal_order%d.pdf"%(order_list[i-1]), dpi=150)

    #absolute plot for parts
    fig9= plt.figure(9)
    axis9 = fig9.gca()
    a12=axis9.bar(1, timeassemblypred_int, width,
                 linewidth=1,color="red")
    b=timeassemblypred_int
    a13=axis9.bar(1, timesolvepred_int, width,
                 linewidth=1,color="blue",bottom=b)

    a14=axis9.bar(2, timeassemblyupd_int, width,
                 linewidth=1,color="red")
    b=timeassemblyupd_int
    a15=axis9.bar(2, timesolveupd_int, width,
                 linewidth=1,color="blue",bottom=b)

    a16=axis9.bar(3, timeassemblycorr_int, width,
                 linewidth=1,color="red")
    b=timeassemblycorr_int
    a17=axis9.bar(3, timesolvecorr_int, width,
                 linewidth=1,color="blue",bottom=b)
    
    axis9.legend((a12[0],a13[0]),
    ("assembly (snes jacobian eval)", "solve (KSP solve)"))
    axis9.set_xlabel('Parts')
    axis9.set_ylabel('Time [s]')
    fig9.savefig("solveassembly_internal/tasparts_absolute_internal_order%d.pdf"%(order_list[i-1]), dpi=150)

    ##########################################################

    #plot for various orders for all absolute
    a18=axis10.bar(order, timeassembly_int, width,
                 linewidth=1,color="red")
    b=timeassembly_int
    a19=axis10.bar(order, timesolve_int, width,
                 linewidth=1,color="blue",bottom=b)

    #plot for various orders for all relative
    time=timeassembly_int+timesolve_int
    a20=axis11.bar(order, timeassembly_int/time, width,
                 linewidth=1,color="red")
    b=timeassembly_int/time
    a21=axis11.bar(order, timesolve_int/time, width,
                 linewidth=1,color="blue",bottom=b)

    return a18,a19,a20,a21



####################################################################
#############GENERAL PERFORMANCE DISTRIBUTION PLOTS################
####################################################################
order_list=[1,2,3]#,4]

#gather all filenames
order_data= ["results/timedata_taylorgreen_ORDER%d_CFL10_RE1_TMAX1_XLEN6_BCdirichlet.csv" % i
              for i in order_list]

#readin all data
readin_data = pd.concat(pd.read_csv(data,nrows=1) for data in order_data)
#readin_data=readin_data.iloc[1,:]#only for one N

#order data by order number
order_group =readin_data.groupby(["order"], as_index=False)

#gather all dofs
dof_list = [e[1] for e in readin_data["sum dofs"].drop_duplicates().items()]#use sum dofs instead

if not os.path.exists(os.path.dirname('distribution_external/')):
            os.makedirs(os.path.dirname('distribution_external/'))
#plot overall percentages
fig1= plt.figure(1)
axis1 = fig1.gca()
axis1.set_ylabel('Normalised Time')

#plot percentages for assembly split by predictor, update, corrector 
fig2= plt.figure(2)
axis2 = fig2.gca()
axis2.set_ylabel('Normalised Time')

#plot internal absolute and relative solve/assembly for all orders
if not os.path.exists(os.path.dirname('solveassembly_internal/')):
            os.makedirs(os.path.dirname('solveassembly_internal/'))

fig10= plt.figure(10)
axis10 = fig10.gca()
axis10.set_ylabel('Absolute Time')

fig11= plt.figure(11)
axis11 = fig11.gca()
axis11.set_ylabel('Normalised Time')

#run through all data
width=0.35
i=1
labels=[]
for data in order_group:
    sum_dof,columns = data

    #gather times for plot of overall time distribution
    timeoverall=columns.taylorgreen
    timeconfig=columns.configuration+columns["spcs configuration"]+columns["initial values"]
    timeforms=columns["build forms"]
    timesolvers=columns["build problems and solvers"]
    timeprog=columns["time progressing"]
    timepost=columns.postprocessing

    a1 = axis1.bar(i,timeconfig/timeoverall, width, linewidth=1,color="blue")
    b=timeconfig/timeoverall
    a2 = axis1.bar(i, timeforms/timeoverall, width,bottom=b,
                linewidth=1,color="red")   
    b+=timeforms/timeoverall
    a3 = axis1.bar(i, timesolvers/timeoverall, width,bottom=b,
                linewidth=1,color="orange") 
    b+=timesolvers/timeoverall
    a4 = axis1.bar(i, timeprog/timeoverall, width,bottom=b,
                linewidth=1,color="yellow")
    b+=timeprog/timeoverall
    a5 = axis1.bar(i, timepost/timeoverall, width,bottom=b,
                linewidth=1,color="green")


    #gather times for plot of assembly split
    timepred=columns["predictor"]
    timeupd=columns["update"]
    timecorr=0

    timeassembly=timepred+timeupd+timecorr
    a6 = axis2.bar(i, timepred/timeassembly, width,
                 linewidth=1,color="red")
    b=timepred/timeassembly
    a7 = axis2.bar(i, timeupd/timeassembly, width,bottom=b,
                linewidth=1,color="blue")
    b+=timeupd/timeassembly
    a8 = axis2.bar(i, timecorr/timeassembly, width,bottom=b,
                linewidth=1,color="green")
    b+=timecorr/timeassembly


    #internal assembly/solve
    a18,a19,a20,a21=solveassembly_internal(columns,axis10,axis11,i)


    labels.append("Order=%d \n (DOF=%d)" % (order_list[i-1],sum_dof))
    i+=1

####internal assembly/solve
axis10.legend((a18[0],a19[0]),
("assembly","solve"))
axis10.set_xlabel("Order")
fig10.savefig("solveassembly_internal/tasall_absolute_internal.pdf", dpi=150)

axis11.legend((a20[0],a21[0]),
("assembly","solve"))
axis11.set_xlabel("Order")
fig11.savefig("solveassembly_internal/tasall_relative_internal.pdf", dpi=150)
#####

####external distributions
#set legends and savefigs
axis1.legend((a1[0],a2[0],a3[0],a4[0],a5[0]),
("config","forms","problems and solver","time stepping","postprocessing"))
fig1.savefig("distribution_external/general_tasAll_external.pdf", dpi=150)

axis2.legend((a6[0],a7[0],a8[0]),
("predictor assembly","update assembly","corrector assembly"))
fig2.savefig("distribution_external/general_tasAssembly_external.pdf", dpi=150)
####

######################## here actually TAS stuff starts

if not os.path.exists(os.path.dirname('tas/')):
            os.makedirs(os.path.dirname('tas/'))

#gather all times
timeoverall_list=[e[1] for e in readin_data["taylorgreen"].items()]

#gather all errors
veloerror_list=[e[1] for e in readin_data["L2Velo"].items()]
preserror_list=[e[1] for e in readin_data["L2Pres"].items()]
error_list=[x+y for x, y in zip(veloerror_list,preserror_list)]

####################################################################
############# MESH CONVERGENCE #####################################
####################################################################

doa=-np.log10(error_list)#digits of accuracy
dos=np.log10(dof_list)#digits of size

fig4= plt.figure(4)
axis4 = fig4.gca()
axis4.set_ylabel('DoA')
axis4.set_xlabel('DoS')

axis4.loglog(dos,doa,"x-",label="DG - GAMG")
axis4.legend()
fig4.savefig("tas/tasMesh.pdf", dpi=150)

####################################################################
#############    STATIC SCALING  ###################################
####################################################################

#static scaling
fig5= plt.figure(5)
axis5 = fig5.gca()
axis5.set_ylabel('DoF/s')
axis5.set_xlabel('Time(s)')

#gather static scaling information
#unknowns per second
dofpersecond=[x/y for x, y in zip(dof_list,timeoverall_list)]
axis5.loglog(timeoverall_list,dofpersecond,"x-",label="DG - GAMG")
axis5.legend()
fig5.savefig("tas/tasStatic.pdf", dpi=150)


####################################################################
############# ACCURACY #############################################
####################################################################
efficacy=[x*y for x, y in zip(error_list,timeoverall_list)]
doe=np.log10(efficacy)#digits of efficacy

fig6= plt.figure(6)
axis6 = fig6.gca()
axis6.set_ylabel('DoE')
axis6.set_xlabel('Time(s)')

axis6.loglog(timeoverall_list,doe,"x-",label="DG - GAMG")
axis6.legend()
fig6.savefig("tas/tasEfficacy.pdf", dpi=150)


####################################################################
############# TRUE STATIC SCALING ##################################
####################################################################

scaling=[x/y for x, y in zip(doa,dos)]
truedofpersecond=[x*y for x, y in zip(scaling,dofpersecond)]

fig8= plt.figure(8)
axis8 = fig8.gca()
axis8.set_ylabel('True DoF/s')
axis8.set_xlabel('Time(s)')

axis8.loglog(timeoverall_list,truedofpersecond,"x-",label="DG - GAMG")
axis8.legend()
fig8.savefig("tas/tasTrueStatic.pdf", dpi=150)