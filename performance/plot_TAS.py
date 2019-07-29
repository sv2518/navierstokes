

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
axis1.set_ylabel('Time [s]')

#plot percentages for assembly split by predictor, update, corrector 
fig2= plt.figure(2)
axis2 = fig2.gca()
axis2.set_ylabel('Time [s]')

#plot percentages split by predictor, update, corrector solve
fig3= plt.figure(3)
axis3 = fig3.gca()
axis3.set_ylabel('Time [s]')

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


