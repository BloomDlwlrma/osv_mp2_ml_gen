from pyscf import gto, scf

mol = gto.M()

mol.atom = '''
O          0.00000000        0.11468200        0.00000000
H          0.75406600       -0.45872600        0.00000000
H         -0.75406600       -0.45872600        0.00000000
'''

mol.basis = "cc-pVTZ"

mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

print(f"Assigned Basis: {mol.basis}")
print("Basis set of O\n", mol._basis['O'])
print("Basis set of H\n", mol._basis['H'])
