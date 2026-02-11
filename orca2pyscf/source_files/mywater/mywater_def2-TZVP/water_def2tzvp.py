from pyscf import gto, scf
from mokit.lib.fch2py import fch2py
from mokit.lib.ortho import check_orthonormal
import numpy as np

mol = gto.M()
# 3 atom(s)
mol.atom = '''
O1          0.00000000        0.11468200        0.00000000
H1          0.75406600       -0.45872600        0.00000000
H2         -0.75406600       -0.45872600        0.00000000
'''

mol.basis = {
'O1': gto.basis.parse('''
O     S 
    0.270323826E+05    0.572273287E-03
    0.405238714E+04    0.443532330E-02
    0.922327227E+03    0.230201076E-01
    0.261240710E+03    0.928224907E-01
    0.853546414E+02    0.293785000E+00
    0.310350352E+02    0.674016045E+00
O     S 
    0.122608607E+02    0.638399370E+00
    0.499870760E+01    0.395345871E+00
O     S 
    0.117031082E+01    0.100000000E+01
O     S 
    0.464747410E+00    0.100000000E+01
O     S 
    0.185045364E+00    0.100000000E+01
O     P 
    0.632749548E+02    0.120183321E-01
    0.146270494E+02    0.830054217E-01
    0.445012235E+01    0.319917439E+00
    0.152757996E+01    0.707155429E+00
O     P 
    0.529351179E+00    0.100000000E+01
O     P 
    0.174784213E+00    0.100000000E+01
O     D 
    0.231400000E+01    0.100000000E+01
O     D 
    0.645000000E+00    0.100000000E+01
O     F 
    0.142800000E+01    0.100000000E+01
'''),
'H1': gto.basis.parse('''
H     S 
    0.340613410E+02    0.254393072E-01
    0.512357460E+01    0.190085949E+00
    0.116466260E+01    0.852441130E+00
H     S 
    0.327230410E+00    0.100000000E+01
H     S 
    0.103072410E+00    0.100000000E+01
H     P 
    0.800000000E+00    0.100000000E+01
'''),
'H2': gto.basis.parse('''
H     S 
    0.340613410E+02    0.254393072E-01
    0.512357460E+01    0.190085949E+00
    0.116466260E+01    0.852441130E+00
H     S 
    0.327230410E+00    0.100000000E+01
H     S 
    0.103072410E+00    0.100000000E+01
H     P 
    0.800000000E+00    0.100000000E+01
''')}

mol.basis = "def2-TZVP"

# Remember to check the charge and spin
mol.charge = 0
mol.spin = 0
mol.verbose = 1
mol.build(parse_arg=False)

nao = mol.nao
mf = scf.RHF(mol).density_fit(auxbasis='def2-tzvp-jkfit')
mf.max_cycle = 1
mf.init_guess = '1e'
# mf.kernel()

# read MOs from .fch(k) file
hf_fch = 'water_def2tzvp.fch'
# nbf = mf.mo_coeff.shape[0]
# nif = mf.mo_coeff.shape[1]
# mf.mo_coeff = fch2py(hf_fch, nbf, nif, 'a')
mf.mo_coeff = fch2py(hf_fch, nao, nao, 'a')
# read done

# check if input MOs are orthonormal
S = mol.intor_symmetric('int1e_ovlp')
check_orthonormal(nao, nao, mf.mo_coeff, S)

no = mol.nelectron // 2
mo_occ = np.zeros(nao)
mo_occ[:no] = 2
mf.mo_occ = mo_occ


dm = mf.make_rdm1()
fock = mf.get_fock()
hcore = mf.get_hcore()
escf = 0.5 * np.sum(dm * (hcore + fock)) + mol.energy_nuc()
print("SCF energy (directly read MOs from ORCA): ", escf)





# mf.max_cycle = 10
# mf.kernel(dm0=dm)
# escf = mf.e_tot
# print("SCF energy (with optimization): ", escf)

print("Numner of auxiliary basis: ", mf.with_df.auxmol.nao_nr())

print("Norm of MOs: ", np.linalg.norm(mf.mo_coeff)) 
print("Norm of DM: ", np.linalg.norm(dm)) 
print("Norm of Fock: ", np.linalg.norm(fock)) 
