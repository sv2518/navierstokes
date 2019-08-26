

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

scaling="_linear_order_scaled"
name="taylorgreen"+scaling
#gather all filenames
cfl_list=[20,10,8,4,2,1,0]
#cfl_list=[50,20,15,10]#,10,8,4,2,1]
cfl_data= ["results/"+name+"_CFL%d_RE1_TMAX1_XLEN6_N6_BCdirichlet.csv" % i
              for i in cfl_list]

#readin all data
readin_data = pd.concat(pd.read_csv(data) for data in cfl_data)

#order data by cfl number
order_group =readin_data.groupby(["CFL"], as_index=False)


#gather all orders
order_list = [e[1] for e in readin_data["Order"].drop_duplicates().items()]

#plot convergence rate
value=0.01#constant for scaling the reference orders
xlabel="p"
fig_velo = plt.figure(1)
fig_pres= plt.figure(2)
axis_velo = fig_velo.gca()
axis_pres= fig_pres.gca()
if scaling=="quadratic":
    axis_velo.semilogy(order_list,value*np.power(order_list[::-1],1)*np.power(order_list[::-1],1),'b-',label="$\propto$ ("+str(xlabel)+"^2)")
    axis_pres.semilogy(order_list,value*np.power(order_list[::-1],1)*np.power(order_list[::-1],1),'b-',label="$\propto$ ("+str(xlabel)+"^2)")
elif scaling=="linear":
    axis_velo.semilogy(order_list,value*np.power(order_list[::-1],1),'b-',label="$\propto$ ("+str(xlabel)+")")
    axis_pres.semilogy(order_list,value*np.power(order_list[::-1],1),'b-',label="$\propto$ ("+str(xlabel)+")")

    
axis_velo.set_xlabel(xlabel)
axis_velo.set_ylabel('$Error$')
axis_pres.set_xlabel(xlabel)
axis_pres.set_ylabel('$Error$')

for data in order_group:
    cfl,columns = data
    print(data)
    label="cfl= "+str(cfl)
    axis_velo.semilogy(order_list,columns.LinfVelo,"x-",label=label)
    axis_pres.semilogy(order_list,columns.LinfPres,"x-",label=label)

    axis_velo.legend()
    axis_pres.legend()

if not os.path.exists(os.path.dirname('plots/')):
    os.makedirs(os.path.dirname('plots/'))
fig_velo.savefig("plots/"+name+"_velo_dirichlet.pdf", dpi=150)
fig_pres.savefig("plots/"+name+"_pres_dirichlet.pdf", dpi=150)

