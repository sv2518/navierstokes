from tests.taylorgreen import *
import matplotlib.pyplot as plt
from helpers.plot_convergence_velo_pres import plot_convergence_velo_pres
import math

#####################MAIN#############################
#plottin spatial convergence for taylor green vortices

error_velo=[]
error_pres=[]
list_dx=[]





RE=2*pi*100

D=2#space dimension
print("d is:",D)

cflList=[1]
u=1
refin=[1,0.5,0.1]#,0.025,0.01]
#increasing spatial refinement (number of elements)
errorList=[]


#monster=[[[0.09330596864195985, 0.0029349957781368806, 0.0009213847526612826], [0.0044647464857818595, 0.003791292417715908, 7.644343640942757e-05], [0.2, 0.1, 0.05]]]
tmp0=[0.023173698442726764, 0.00848042110675954, 0.003711205018427097, 0.0014341506033510002, 0.0005381226606168856]
tmp0=[1.0017835759270675, 0.0759705838087938, 0.030803525340143943]
for cc in range(0,len(cflList)):
        #plot_convergence_velo_pres(monster[cc][0],monster[cc][1],refin,0.1,"hi")
       plot_convergence_velo_pres(tmp0,tmp0,refin,0.1,"hi")
plt.show()
for cfl in cflList:
    error_velo=[]
    error_pres=[]
    for dt in refin:
        #time stepping
        u=1
        print(u)
        nue=1
        dx=2*u*0.0075/(10/D**2)
        print(dx)
        T=2/dt
        t=[dt,T]
        #solve
        w,err_u,err_p,dx = taylorgreen(dx, D,t,RE,True)
        u,p=w.split()
        error_velo.append(err_u)
        error_pres.append(err_p)
        list_dx.append(dx)

        print(error_velo)    
    errorList.append([error_velo,error_pres,list_dx])

#convergence plots
print(errorList)

for cc in range(0,len(cflList)):
    plot_convergence_velo_pres(errorList[cc][0],errorList[cc][1],refin,1,"hi")
plt.show()

#10,25,30,50


#1,2,3,5
monster=[[[1.0846916551828996, 1.024491381082698, 0.962360976327584, 0.8351911601092783, 0.47422306026835975, 0.3428080404200763, 0.12854229446494614, 0.08509862536224028, 0.05564853194674173], [0.4936530096840632, 0.48103797223815503, 0.4673007193293223, 0.4369104733771959, 0.3338394732711285, 0.2543944432390903, 0.19975464913965857, 0.19312387798701194, 0.13253626938619592], [0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.04]], [[1.0846916551828998, 1.0244913810826983, 0.962360976327584, 0.8351911601092783, 0.47422306026835975, 0.3428080404200763, 0.12854229446494614, 0.08509862536224028, 0.055648531946741735], [0.49365300968406317, 0.48103797223815514, 0.4673007193293225, 0.43691047337719585, 0.3338394732711285, 0.2543944432390903, 0.19975464913965857, 0.19312387798701194, 0.13253626938619592], [0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.04]], [[1.0846916551828998, 1.0244913810826979, 0.9623609763275841, 0.835191160109278, 0.47422306026835975, 0.3428080404200763, 0.1285422944649461, 0.08509862536224028, 0.05564853194674177], [0.4936530096840632, 0.48103797223815514, 0.46730071932932243, 0.4369104733771959, 0.3338394732711285, 0.2543944432390903, 0.1997546491396585, 0.19312387798701194, 0.1325362693861959], [0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.04]], [[1.084691655182901, 1.0244913810826983, 0.962360976327584, 0.8351911601092784, 0.47422306026835975, 0.3428080404200763, 0.12854229446494614, 0.08509862536224028, 0.05650795416999576], [0.4936530096840633, 0.48103797223815514, 0.46730071932932227, 0.43691047337719585, 0.3338394732711285, 0.2543944432390903, 0.19975464913965857, 0.19312387798701194, 0.13395461387310414], [0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.044721359549995794, 0.11832159566199232, 0.1140175425099138, 0.10954451150103321, 0.1, 0.08944271909999159, 0.07745966692414834, 0.06324555320336758, 0.054772255750516606, 0.04]]]
for cc in range(0,len(cflList)):
        plot_convergence_velo_pres(monster[cc][0],monster[cc][1],refin,1,"hi")
plt.show()


#plot actual errors in L2
#plt.plot(refin,np.array([3.9,1.2,0.33,0.093,0.026]),label="paper")
#plt.plot(refin,np.array(error_pres)/100,label="mine/100")
#plt.legend()
#plt.title("L2 error pressure")
#plt.show()

#plt.plot(refin,np.array([0.23,0.062,0.016,0.0042,0.0011]),label="paper")
#plt.plot(refin,error_velo,label="mine")
#plt.legend()
#plt.title("L2 error velocity")
#plt.show()
