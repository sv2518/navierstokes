

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


####################################################################
#############GENERAL PERFORMANCE DISTRIBUTION PLOTS################
####################################################################
order_list=[1,2,3,4]

#gather all filenames
order_data= ["results/timedata_taylorgreen_ORDER%dCFL20_RE1_TMAX1_XLEN6_N6_BCdirichlet.csv" % i
              for i in order_list]

#readin all data
readin_data = pd.concat(pd.read_csv(data) for data in order_data)

#order data by order number
order_group =readin_data.groupby(["sum dofs"], as_index=False)

#gather all dofs
dof_list = [e[1] for e in readin_data["sum dofs"].drop_duplicates().items()]#use sum dofs instead


#plot overall percentages
fig1= plt.figure(1)
axis1 = fig1.gca()
axis1.set_ylabel('Normalised Time')

#plot percentages for assembly split by predictor, update, corrector 
fig2= plt.figure(2)
axis2 = fig2.gca()
axis2.set_ylabel('Normalised Time')

#plot percentages split by predictor, update, corrector solve
fig3= plt.figure(3)
axis3 = fig3.gca()
axis3.set_ylabel('Normalised Time')

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


    #gather times for solve splits
    timepredsolve=columns["predictor solve"]
    timeupdsolve=columns["update solve"]
    timecorrsolve=columns["corrector solve"]

    timesolve=timepredsolve+timeupdsolve+timecorrsolve
    a9 = axis3.bar(i, timepredsolve/timesolve, width,
                 linewidth=1,color="red")
    b=timepredsolve/timesolve
    a10 = axis3.bar(i, timeupdsolve/timesolve, width,bottom=b,
                linewidth=1,color="blue")
    b+=timeupdsolve/timesolve
    a11 = axis3.bar(i, timecorrsolve/timesolve, width,bottom=b,
                linewidth=1,color="green")




    labels.append("Order=%d \n (DOF=%d)" % (order_list[i-1],sum_dof))
    i+=1




#set legends and savefigs
axis1.legend((a1[0],a2[0],a3[0],a4[0],a5[0]),
("config","assembly forms","assembly problems and solver","time stepping","postprocessing"))
fig1.savefig("tasAll.pdf", dpi=150)

axis2.legend((a6[0],a7[0],a8[0]),
("predictor assembly","update assembly","corrector assembly"))
fig2.savefig("tasAssembly.pdf", dpi=150)

axis3.legend((a9[0],a10[0],a11[0]),
("predictor solve","update solve","corrector solve"))
fig3.savefig("tasSolve.pdf", dpi=150)

########################

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
fig4.savefig("tasMesh.pdf", dpi=150)

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
fig5.savefig("tasStatic.pdf", dpi=150)


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
fig6.savefig("tasEfficacy.pdf", dpi=150)


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
fig8.savefig("tasTrueStatic.pdf", dpi=150)