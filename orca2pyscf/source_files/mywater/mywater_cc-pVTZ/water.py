from pyscf import gto, scf
from mokit.lib.fch2py import fch2py
from mokit.lib.ortho import check_orthonormal

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
    0.153300000E+05    0.520198307E-03
    0.229900000E+04    0.402334478E-02
    0.522400000E+03    0.207290833E-01
    0.147300000E+03    0.810823271E-01
    0.475500000E+02    0.236226352E+00
    0.167600000E+02    0.443518209E+00
    0.620700000E+01    0.358670589E+00
    0.688200000E+00   -0.834979660E-02
O     S 
    0.153300000E+05   -0.197236012E-03
    0.229900000E+04   -0.153501070E-02
    0.522400000E+03   -0.795118391E-02
    0.147300000E+03   -0.321134529E-01
    0.475500000E+02   -0.100269643E+00
    0.167600000E+02   -0.234047112E+00
    0.620700000E+01   -0.301410928E+00
    0.688200000E+00    0.103491965E+01
O     S 
    0.175200000E+01    0.100000000E+01
O     S 
    0.238400000E+00    0.100000000E+01
O     P 
    0.344600000E+02    0.411634896E-01
    0.774900000E+01    0.257762836E+00
    0.228000000E+01    0.802419275E+00
O     P 
    0.715600000E+00    0.100000000E+01
O     P 
    0.214000000E+00    0.100000000E+01
O     D 
    0.231400000E+01    0.100000000E+01
O     D 
    0.645000000E+00    0.100000000E+01
O     F 
    0.142800000E+01    0.100000000E+01
'''),
'H1': gto.basis.parse('''
H     S 
    0.338700000E+02    0.254948633E-01
    0.509500000E+01    0.190362766E+00
    0.115900000E+01    0.852162022E+00
H     S 
    0.325800000E+00    0.100000000E+01
H     S 
    0.102700000E+00    0.100000000E+01
H     P 
    0.140700000E+01    0.100000000E+01
H     P 
    0.388000000E+00    0.100000000E+01
H     D 
    0.105700000E+01    0.100000000E+01
'''),
'H2': gto.basis.parse('''
H     S 
    0.338700000E+02    0.254948633E-01
    0.509500000E+01    0.190362766E+00
    0.115900000E+01    0.852162022E+00
H     S 
    0.325800000E+00    0.100000000E+01
H     S 
    0.102700000E+00    0.100000000E+01
H     P 
    0.140700000E+01    0.100000000E+01
H     P 
    0.388000000E+00    0.100000000E+01
H     D 
    0.105700000E+01    0.100000000E+01
''')}

# Remember to check the charge and spin
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build(parse_arg=False)

mf = scf.RHF(mol)
mf.max_cycle = 1
mf.init_guess = '1e'
mf.kernel()

# read MOs from .fch(k) file
hf_fch = 'water.fch'
nbf = mf.mo_coeff.shape[0]
nif = mf.mo_coeff.shape[1]
mf.mo_coeff = fch2py(hf_fch, nbf, nif, 'a')
# read done

# check if input MOs are orthonormal
S = mol.intor_symmetric('int1e_ovlp')
check_orthonormal(nbf, nif, mf.mo_coeff, S)

#dm = mf.make_rdm1()
#mf.max_cycle = 10
#mf.kernel(dm0=dm)

