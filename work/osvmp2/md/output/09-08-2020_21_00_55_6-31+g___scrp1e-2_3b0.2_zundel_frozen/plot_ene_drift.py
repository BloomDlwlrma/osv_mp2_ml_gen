import sys
import numpy as np
import matplotlib.pyplot as plt


def get_temp(e_k, natom):
    #T = e_k*2/(N_dof*kb)
    #N_dof = 3natom - Nc - Ncomm: degree of freedom of the system
    #Boltzmann constant in eV/K: 8.617333262e-5 eV/K
    if fix_atom:
        N_dof = 3*(natom-1)
    elif fix_com:
        N_dof = 3*natom - 3
    else:
        N_dof = 3*natom

    kb = 8.617333262e-5
    return e_k*2/(N_dof*kb)
def get_drift(ene_list):
    x = np.arange(len(ene_list))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, ene_list, rcond=None)[0]
    e0 = m*x[0]+c
    e1 = m*x[-1]+c
    return (e1-e0)

output = sys.argv[1]
natom = 19
fix_atom = False#True
fix_com = True
temp_list = []
ene_kin = []
ene_list = []
with open(output, 'r') as f:
    lines = f.readlines()
    for l in lines[6:]:
        #temp_list.append(float(l.split()[3]))
        ene_kin.append(float(l.split()[-2]))
        if fix_atom:
            ncount = (natom-1)
        else:
            ncount = natom
        ene_list.append(96.487 * (float(l.split()[-1])/natom + float(l.split()[-2])/ncount))
ene_list = np.asarray(ene_list[1:]) 
for e_k in ene_kin:
    temp_i = get_temp(e_k, natom)
    temp_list.append(temp_i)
    print(temp_i)
print("Average temperature: %.2f K"%np.mean(temp_list))
print("Energy drift: %.4f kj/mol"%get_drift(ene_list))
plt.plot(np.arange(len(ene_list)), ene_list-min(ene_list),  label='Original data')
#plt.ylim(0,5)
plt.legend()
plt.show()
