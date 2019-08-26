

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
sns.set_palette("deep")
current_palette = sns.color_palette()
tips =['o','v','s','P','*',"D",2]
#order is not order it is the nth specification of dof numbers
def solveassembly_internal(columns,axis10,axis11,jp1,dof,order,case):

    #gather internal data
    timeassemblypred_int=columns["jac_eval_time_pred"]
    timesolvepred_int=columns["pc_apply_time_pred"]
    timeassemblyupd_int=columns["HDGAssembly"]
    timesolveupd_int=columns["pc_apply_time_upd"]
    timeassemblycorr_int=columns["jac_eval_time_corr"]
    timesolvecorr_int=columns["pc_apply_time_corr"]

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
                 linewidth=1,color=current_palette[9])
    b=timeassemblypred_int/timepred_int
    a13=axis9.bar(1, timesolvepred_int/timepred_int, width,
                 linewidth=1,color=current_palette[5],bottom=b)

    a14=axis9.bar(2, timeassemblyupd_int/timeupd_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassemblyupd_int/timeupd_int
    a15=axis9.bar(2, timesolveupd_int/timeupd_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)

    a16=axis9.bar(3, timeassemblycorr_int/timecorr_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassemblycorr_int/timecorr_int
    a17=axis9.bar(3, timesolvecorr_int/timecorr_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)
    
    axis9.legend((a12[0],a13[0]),
    ("assembly (snes jacobian eval/hybrid operator assembly)", "solve (PC apply)"))
    axis9.set_xlabel('Parts')
    axis9.set_ylabel('Normalised Time')
    fig9.savefig('solveassembly_internal/'+case+'/tasparts_relative_internal_order%d_dof%d.pdf'%(order,dof), dpi=150)

    #absolute plot for parts
    fig99= plt.figure(99)
    axis99 = fig99.gca()
    a12=axis99.bar(1, timeassemblypred_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassemblypred_int
    a13=axis99.bar(1, timesolvepred_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)

    a14=axis99.bar(2, timeassemblyupd_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassemblyupd_int
    a15=axis99.bar(2, timesolveupd_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)

    a16=axis99.bar(3, timeassemblycorr_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassemblycorr_int
    a17=axis99.bar(3, timesolvecorr_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)
    
    axis99.legend((a12[0],a13[0]),
    ("assembly (snes jacobian eval/hybrid operator assembly)", "solve (PC apply)"))
    axis99.set_xlabel('Parts')
    axis99.set_ylabel('Time [s]')
    fig99.savefig('solveassembly_internal/'+case+'/tasparts_absolute_internal_order%d_dof%d.pdf'%(order,dof), dpi=150)

    ##########################################################

    #plot for various orders for all absolute
    a18=axis10.bar(jp1, timeassembly_int, width,
                 linewidth=1,color=current_palette[9])
    b=timeassembly_int
    a19=axis10.bar(jp1, timesolve_int, width,
                 linewidth=1,color=current_palette[3],bottom=b)

    #plot for various jp1s for all relative
    time=timeassembly_int+timesolve_int
    a20=axis11.bar(jp1, timeassembly_int/time, width,
                 linewidth=1,color=current_palette[9])
    b=timeassembly_int/time
    a21=axis11.bar(jp1, timesolve_int/time, width,
                 linewidth=1,color=current_palette[3],bottom=b)

    return a18,a19,a20,a21,time

def tas_spectrum(error_list,dof_group_list,timeoverall_list,case,sol):

    plt.close(4)
    plt.close(5)
    plt.close(6)
    plt.close(8)
    ####################################################################
    ############# MESH CONVERGENCE #####################################
    ####################################################################

    doa=[-np.log10(error) for error in error_list]#digits of accuracy
    dos=[np.log10(np.sqrt(dof)) for dof in dof_group_list]#digits of size

    fig4= plt.figure(4)
    axis4 = fig4.gca()
    axis4.set_ylabel('DoA')
    axis4.set_xlabel('DoS')

    for i,order in enumerate(order_list):
        axis4.plot(dos[i],doa[i],"x-",label="DG%d - GAMG"%order, marker=tips[i])

    axis4.legend()
    axis4.grid()
    fig4.savefig('tas/'+case+'/tasMesh_'+sol+'.pdf', dpi=150)
    ####################################################################
    #############    STATIC SCALING  ###################################
    ####################################################################

    #static scaling
    fig5= plt.figure(5)
    axis5 = fig5.gca()
    axis5.set_ylabel('DoF/s')
    axis5.set_xlabel('Time(s)')

    #ggather static scaling information
    #unknowns per second
    dofpersecond=[]
    for i,order in enumerate(order_list):
        dofpersecond.append([x*TMAX/z/y for x, y,z in zip(dof_group_list[i],timeoverall_list[i],dt_list[i])])
        axis5.loglog(timeoverall_list[i],dofpersecond[i],"x-",label="DG%d - GAMG"%order, marker=tips[i])

    axis5.legend()
    axis5.grid()
    fig5.savefig('tas/'+case+'/tasStatic_'+sol+'.pdf', dpi=150)

    ####################################################################
    ############# ACCURACY #############################################
    ####################################################################

    fig6= plt.figure(6)
    axis6 = fig6.gca()
    axis6.set_ylabel('DoE')
    axis6.set_xlabel('Time(s)')


    for i,order in enumerate(order_list):
        efficacy=[x*y for x, y in zip(error_list[i],timeoverall_list[i])]
        doe=-np.log10(efficacy)#digits of efficacy
        axis6.plot(np.log10(timeoverall_list[i]),doe,"x-",label="DG%d - GAMG"%order, marker=tips[i])

    axis6.legend()
    axis6.grid()
    fig6.savefig('tas/'+case+'/tasEfficacy_'+sol+'.pdf', dpi=150)

    ####################################################################
    ############# TRUE STATIC SCALING ##################################
    ####################################################################

    fig8= plt.figure(8)
    axis8 = fig8.gca()
    axis8.set_ylabel('True DoF/s')
    axis8.set_xlabel('Time(s)')

    for i,order in enumerate(order_list):
        scaling=[x/y for x, y in zip(doa[i],dos[i])]
        truedofpersecond=[x*y for x, y in zip(scaling,dofpersecond[i])]
        axis8.loglog(np.log10(timeoverall_list[i]),truedofpersecond,"x-",label="DG - GAMG", marker=tips[i])
    axis8.legend()
    axis8.grid()
    fig8.savefig('tas/'+case+'/tasTrueStatic_'+sol+'.pdf', dpi=150)

def convergence_rates(error_list,dof_list,order_list,type):
    rows_conv_rate=[]
    for i,order in enumerate(order_list):
        one_error_list=error_list[i]
        one_dof_list=dof_group_list[i]
        conv_rate=[]
        conv_rate.append("DG%d"%(i+1))
        names=[0]
        for i,error in enumerate(one_error_list):
            if i<len(one_error_list)-1:
                conv_rate.append(np.log10(one_error_list[i]/one_error_list[i+1])/np.log10(np.sqrt(one_dof_list[i+1])/np.sqrt(one_dof_list[i])))
            else:
                conv_rate.append('-')
            names.append(one_dof_list[i])
        rows_conv_rate.append(conv_rate)

    datafile = pd.DataFrame(rows_conv_rate,columns=names).sort_index(axis=1)   
    result="convergence/"+case+"/timedata_taylorgreen_convergence"+type+".csv"
    if not os.path.exists(os.path.dirname("convergence/"+case+"/")):
        os.makedirs(os.path.dirname("convergence/"+case+"/"))
    datafile.to_csv(result, index=False,mode="w", header=True)


####################################################################
#############GENERAL PERFORMANCE DISTRIBUTION PLOTS################
####################################################################
order_list=[1,2,3,4,5]
dofs_list=[10000,20000,30000,40000,50000,80000,100000,200000,400000,600000,800000]
tmax='1e-9'
TMAX=float(tmax)
case='gamg_TMAX_'+tmax


#gather all filenames
order_data=[]
for order in order_list:
    #dof_data= ["results/timedata_taylorgreen_ORDER%d_CFL10_RE1_TMAX1_XLEN6_BCperiodic_DOFS%d.csv" % (order,i)
    #          for i in dofs_list]
    dof_data= ['results/'+case+'/timedata_taylorgreen_ORDER%d_CFL0_RE1_TMAX0_XLEN6_BCdirichlet_DOFS%d_PRECONgamg_STABSlinear.csv' % (order,i)
              for i in dofs_list]
    order_data.append(dof_data)

#readin all data
order_group_data=[]
dof_group_list=[]
timeoverall_list=[]
veloerror_list=[]
preserror_list=[]
dt_list=[]
for i,order in enumerate(order_list):
    dof_group_data = pd.concat(pd.read_csv(dof_data,nrows=1) for dof_data in order_data[i])
    
    #order data by dof number and append to group by order list
    order_group_data.append(dof_group_data.groupby(["sum dofs"], as_index=False))

    #gather all dofs
    dof_group_list.append([e[1] for e in dof_group_data["sum dofs"].items()])#use sum dofs instead

    #gather all times, decide here which times to use!
    timeoverall_list.append([e[1] for e in dof_group_data["taylorgreen"].items()])
    #timeoverall_list=TIME_LIST

    #gather all errors
    veloerror_list.append([e[1] for e in dof_group_data["L2Velo"].items()])
    preserror_list.append([e[1] for e in dof_group_data["L2Pres"].items()])
    dt_list.append([e[1] for e in dof_group_data["dt"].items()])


convergence_rates(veloerror_list,dof_group_list,order_list,"velo")
convergence_rates(preserror_list,dof_group_list,order_list,"pres")


if not os.path.exists(os.path.dirname('distribution_external/'+case+'/')):
    os.makedirs(os.path.dirname('distribution_external/'+case+'/'))

#plot overall percentages
fig1= plt.figure(1)
axis1 = fig1.gca()
axis1.set_ylabel('Normalised Time')

#plot percentages for assembly split by predictor, update, corrector 
fig2= plt.figure(2)
axis2 = fig2.gca()
axis2.set_ylabel('Normalised Time')

#plot internal absolute and relative solve/assembly for all orders
if not os.path.exists(os.path.dirname('solveassembly_internal/'+case+'/')):
    os.makedirs(os.path.dirname('solveassembly_internal/'+case+'/'))

fig10= plt.figure(10)
axis10 = fig10.gca()
axis10.set_ylabel('Absolute Time [s]')
fig11= plt.figure(11)
axis11 = fig11.gca()
axis11.set_ylabel('Normalised Time')

#run through all data
width=0.5
i=1
TIME_LIST=[]
for i,order in enumerate(order_group_data):
    labels=[]
    labels.append(" ")
    for j,data in enumerate(order_group_data[i]):
        sum_dof,columns = data

        #gather times for plot of overall time distribution
        timeoverall=columns.taylorgreen
        timeconfig=columns.configuration+columns["spcs configuration"]+columns["initial values"]
        timeforms=columns["build forms"]
        timesolvers=columns["build problems and solvers"]
        timeprog=columns["time progressing"]
        timepost=columns.postprocessing

        a1 = axis1.bar(j+1,timeconfig/timeoverall, width, linewidth=1,color=current_palette[3])
        b=timeconfig/timeoverall
        a2 = axis1.bar(j+1, timeforms/timeoverall, width,bottom=b,
                    linewidth=1,color=current_palette[9])   
        b+=timeforms/timeoverall
        a3 = axis1.bar(j+1, timesolvers/timeoverall, width,bottom=b,
                    linewidth=1,color="orange") 
        b+=timesolvers/timeoverall
        a4 = axis1.bar(j+1, timeprog/timeoverall, width,bottom=b,
                    linewidth=1,color="yellow")
        b+=timeprog/timeoverall
        a5 = axis1.bar(j+1, timepost/timeoverall, width,bottom=b,
                    linewidth=1,color="green")


        #gather times for plot of assembly split
        timepred=columns["predictor"]
        timeupd=columns["update"]
        timecorr=0

        timeassembly=timepred+timeupd+timecorr
        a6 = axis2.bar(j+1, timepred/timeassembly, width,
                    linewidth=1,color=current_palette[9])
        b=timepred/timeassembly
        a7 = axis2.bar(j+1, timeupd/timeassembly, width,bottom=b,
                    linewidth=1,color=current_palette[3])
        b+=timeupd/timeassembly
        a8 = axis2.bar(j+1, timecorr/timeassembly, width,bottom=b,
                    linewidth=1,color="green")
        b+=timecorr/timeassembly


        #internal assembly/solve
        a18,a19,a20,a21,TIME=solveassembly_internal(columns,axis10,axis11,j+1,sum_dof,i+1,case)
        TIME_LIST.append(int(TIME))
        labels.append("%d" % (dof_group_list[i][j]))

    ####internal assembly/solve
    axis10.legend((a18[0],a19[0]),
    ("assembly","solve"))
    axis10.set_xlabel("DOFS")
    print(labels)
    axis10.set_xticks(np.arange(0, len(labels), 1.0))
    axis10.set_xticklabels(labels)
    fig10.savefig('solveassembly_internal/'+case+'/tasall_absolute_internal_order%d.pdf'%order_list[i], dpi=150)

    axis11.legend((a20[0],a21[0]),
    ("assembly","solve"))
    axis11.set_xlabel("DOFS")
    axis11.set_xticklabels(labels)
    fig11.savefig('solveassembly_internal/'+case+'/tasall_relative_internal_order%d.pdf'%order_list[i], dpi=150)
    #####

    ####external distributions
    #set legends and savefigs
    axis1.legend((a1[0],a2[0],a3[0],a4[0],a5[0]),
    ("config","forms","problems and solver","time stepping","postprocessing"))
    fig1.savefig('distribution_external/'+case+'/general_tasAll_external_order%d.pdf'%order_list[i], dpi=150)

    axis2.legend((a6[0],a7[0],a8[0]),
    ("predictor assembly","update assembly","corrector assembly"))
    fig2.savefig('distribution_external/'+case+'/general_tasAssembly_external_order%d.pdf'%order_list[i], dpi=150)
    ####

################ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####################
################ here actually TAS stuff starts ####################
################ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####################

if not os.path.exists(os.path.dirname('tas/'+case+'/')):
    os.makedirs(os.path.dirname('tas/'+case+'/'))

#plot tas spectrum for velocity
tas_spectrum(veloerror_list,dof_group_list,timeoverall_list,case,"velo")

#plot tas spectrum for pressure
tas_spectrum(preserror_list,dof_group_list,timeoverall_list,case,"pres")
