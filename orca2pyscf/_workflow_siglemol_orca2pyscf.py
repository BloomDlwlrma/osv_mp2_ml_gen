# To reuse legacy OSVMP2 IO of building molecules, which
# 1. Gets coordinates from xyz files
# 2. Set basis sets and other parameters
# 3. Loads HF and localized orbitals from chkfiles
#

# Since basis sets (cc-pVTZ and def2-TZVP) and auxiliary basis sets (cc-pVTZ-jkfit and def2-TZVP-jkfit) used by ORCA and PySCF are exactly the same
# This script should and only should do the following thins:
# 1. Read HF and localized MOs from ORCA and dot the transformation, so the AO orders are corret
# 2. Save HF and localized orbitals in chkfiles with pre-defined format, so OSVMP2 codes could restore calculations with chkfiles


# Therefore, the overall prodecure should be like this:
# 1. always use seperated xyz files for ORCA calculations, so that our OSVMP2 codes could also use them.

# In this script, parse basis set from ORCA files

# In this script, load molecules and read MOs

# In this script, save HF and localized chkfiles


import os 
import sys
import h5py
import subprocess
import numpy as np
from pathlib import Path
from pyscf import gto, scf
from pyscf.df import addons, DF
from mokit.lib.fch2py import fch2py
from mokit.lib.ortho import check_orthonormal
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert ORCA MKL files to PySCF chkfiles")
    parser.add_argument("--mode", type=str, default="test", help="Mode of operation: verify or test")
    args = parser.parse_args()
    return args

def get_atoms(mol):
    atom_list = []
    for ai in mol._atom:
        atom_list.append(ai[0])
    return atom_list


def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
      lines = f.readlines()
      natom = int(lines[0])
      coord = ""
      for l in lines[2:2+natom]:
         if ":" in l:
            lsplit = l.split()
            if lsplit[0][-1] == ":":
               ia_sym_old = lsplit[0]
               ia_atom = ia_sym_old.replace(":")
               ia_sym_new = f"X:{ia_atom}"
               l = l.replace(ia_sym_old, ia_sym_new)
         coord += l
    return natom, coord

def get_M_orca(coord, basis, verbose):
    mybasis = [i.lower().replace('-', '') for i in mybasis]

    if not basis.lower().replace('-', '') in mybasis:
        raise NotImplementedError(f"ORCA basis set {basis} not available yet")

    mole = gto.Mole()
    mole.atom = coord
    mole.basis = basis
    mole.charge = int(os.environ.get("charge", 0))
    mole.spin = int(os.environ.get("spin", 0))
    mole.build(verbose=verbose)

    return mole

def get_aux_orca(mol, basis, mp2fit=False):
    verified_hf_auxbasis = ["cc-pVTZ-jkfit", "def2-TZVP-jkfit"]
    verified_mp2_auxbasis = ["cc-pVTZ-ri", "def2-TZVP-ri"]
    basis = basis.lower().replace('-', '')

    if basis in ["631g", "631g*"]:
        basis = "def2-svp"

    if mp2fit:
        fit_type ="ri"
    else:
        fit_type = "jkfit"

    auxbasis = basis + fit_type

    if not auxbasis in [ibas.lower().replace('-', '') for ibas in verified_hf_auxbasis+verified_mp2_auxbasis]:
        raise NotImplementedError(f"Auxiliary basis {auxbasis} not available")

    auxbasis_dic = {}

    atom_list = get_atoms(mol)
    for atm in set(atom_list):
        auxbasis_dic[atm] = auxbasis
    return auxbasis_dic



def get_mole(mol):
    if mol == None:
        natom, coord = read_xyz(sys.argv[1])
        basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
        use_ecp = bool(int(os.environ.get("use_ecp", 0)))
        basis_molpro = bool(int(os.environ.get("basis_molpro", 0)))
        basis_orca = bool(int(os.environ.get("basis_orca", 0)))
        verbose = 3
        if basis_molpro:
            pass
        elif basis_orca:
            print("Building molecule with ORCA basis set")
            mole = get_M_orca(coord, basis, verbose)
         
        elif use_ecp or "Be" in coord:
            pass
        else:
            mole = gto.M()
            mole.atom = coord
            mole.basis = basis
            mole.charge = int(os.environ.get("charge", 0))
            mole.spin = int(os.environ.get("spin", 0))
            mole.build(verbose=verbose)
            mole.opt_cycle = 0
    else:
        mole = mol
        mole.opt_cycle += 1
    return mole


def make_df(mol):
    auxbasis_hf = get_aux_orca(mol, mol.basis)
    auxbasis_mp2 = get_aux_orca(mol, mol.basis, mp2fit=True)

    auxmol_hf = addons.make_auxmol(mol, auxbasis_hf)
    auxmol_mp2 = addons.make_auxmol(mol, auxbasis_mp2)

    print("Number of AOs: ", mol.nao)
    print("HF density fitting auxiliarty basis: ", auxbasis_hf)
    print("MP2 density fitting auxiliarty basis: ", auxbasis_mp2)
    print("Number of HF auxiliary basis set: ", auxmol_hf.nao_nr())
    print("Number of MP2 auxiliary basis set: ", auxmol_mp2.nao_nr())

    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxbasis = auxbasis_hf
    mf.with_df.auxmol = auxmol_hf
    return mf

def mkl2fch(mklfile, path_mkl2fch):
    '''
    mklfile: path of ORCA generated MKL file
        use `orca_2mkl basename -mkl` to convert basename.gbw to MKL file
    path_mkl2fch: path of executable binary `mkl2fch` provided by MOKIT 
    '''
    if not Path(mklfile).is_file():
        raise ValueError(f"MKL file {mklfile} not a valid file")
    if not Path(path_mkl2fch).is_file():
        raise ValueError(f"mkl2fchk binary {path_mkl2fch} not valid")

    subprocess.run([path_mkl2fch, mklfile],
                   check=True)



def load_orca_mo(mf, mklfile, path_mkl2fch):
    
    mkl2fch(mklfile, path_mkl2fch)
    fchfile = Path(mklfile).with_suffix(".fch")
    nao = mf.mol.nao

    mf.mo_coeff = fch2py(fchfile, nao, nao, 'a')
    ovlp = mf.mol.intor_symmetric('int1e_ovlp')
    check_orthonormal(nao, nao, mf.mo_coeff, ovlp)

    no = mf.mol.nelectron // 2
    mo_occ = np.zeros(nao, dtype="int")
    mo_occ[:no] = 2
    mf.mo_occ = mo_occ

    mf.dm = mf.make_rdm1()
    mf.fock = mf.get_fock()
    hcore = mf.get_hcore()
    mf.escf = 0.5 * np.sum(mf.dm * (hcore + mf.fock)) + mf.mol.energy_nuc()
    print("SCF energy (directly read MOs from ORCA): ", mf.escf)

    

def access_chkfile(chkfile, mode, arrays, cycle=None):
    #The order of the buffer has to be: dm, mo_energy, mo_coeff, mo_occ, mocc, e_tot
    key_list = ["dm", "mo_energy", "mo_coeff", "mo_occ", "mocc", "e_tot"]
    array_dic = {}
    for idx, key_i in enumerate(key_list):
        array_dic[key_i] = arrays[idx]
    with h5py.File(chkfile, mode) as f:
        if mode == 'w':
            for idx, key_i in enumerate(key_list):
                f.create_dataset("scf/%s"%key_i, data=array_dic[key_i])
        else:
            keys_file = f["scf"].keys()
            if mode == 'r+':
                for idx, key_i in enumerate(key_list):
                    if key_i in keys_file:
                        f["scf/%s"%key_i].write_direct(array_dic[key_i])
                    else:
                        f.create_dataset("scf/%s"%key_i, data=array_dic[key_i])
            elif mode == 'r':
                nochk_list = []
                for idx, key_i in enumerate(key_list):
                    if array_dic[key_i] is None:
                        continue
                    if key_i in keys_file:
                        f["scf/%s"%key_i].read_direct(array_dic[key_i])
                    else:
                        #dm, mocc
                        nochk_list.append(key_i)
                for key_i in nochk_list:
                    if key_i == "dm":
                        array_dic[key_i][:] = make_rdm1(array_dic["mo_coeff"], array_dic["mo_occ"])

    if mode == 'r':
        return arrays

def save_chkhf(mf):
    mf.mo_energy = np.diag(
            np.dot(mf.mo_coeff.T, np.dot(mf.fock, mf.mo_coeff))
            )

    mf.mo_occ = mf.get_occ(mf.mo_energy, mf.mo_coeff)

    mf.mocc = mf.mo_coeff[:, mf.mo_occ>0] * (mf.mo_occ[mf.mo_occ>0]**0.5)

    print("Molecular orbital energies")
    print(mf.mo_energy)

    print("Orbital occupation")
    print(mf.mo_occ)

    hfe = np.array([mf.escf])
    chkhf = 'hf_mat.chk'
    access_chkfile(chkhf, 'w', [mf.dm, mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.mocc, hfe])


    # dm, mo_energy, mo_coeff, mo_occ, mocc, hfe

def load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch):
    mkl2fch(mklfile_loc, path_mkl2fch)
    fchfile = Path(mklfile_loc).with_suffix(".fch")
    nao = mf.mol.nao

    mf.o = fch2py(fchfile, nao, nao, 'a')
    ovlp = mf.mol.intor_symmetric('int1e_ovlp')
    check_orthonormal(nao, nao, mf.o, ovlp)


    no = mf.mol.nelectron // 2
    mo_occ = np.zeros(nao, dtype="int")
    mo_occ[:no] = 2
    mf.mo_occ = mo_occ

    mf.dm = mf.make_rdm1(mo_coeff=mf.o)
    mf.fock = mf.get_fock()
    hcore = mf.get_hcore()
    mf.escf = 0.5 * np.sum(mf.dm * (hcore + mf.fock)) + mf.mol.energy_nuc()

    # Get localized Fock matrix in MO space
    mf.o = mf.o[:, mf.mo_occ >0]
    mf.uo = np.dot(mf.mo_coeff[:, mf.mo_occ >0].T, np.dot(ovlp, mf.o))

    uo_res = np.allclose(np.dot(mf.mo_coeff[:, mf.mo_occ>0], mf.uo), mf.o, atol=1e-6)
    print("Checking uo (relaxing constraints a little bit because of numerical accuracy): ", uo_res)
    mf.loc_fock = np.dot(mf.uo.T, 
      np.dot(
          np.dot(mf.mo_coeff.T, np.dot(mf.fock, mf.mo_coeff))[:no, :no],
        mf.uo))
    mf.eo = np.diag(mf.loc_fock)
    print("Localized orbital `energies`\n", mf.eo)


def save_chkloc(mf):
    chkfile_loc = "loc_var.chk"
    with h5py.File(chkfile_loc, 'w') as f:
        f.create_dataset("uo", data=mf.uo)
        f.create_dataset("o", data=mf.o)
        f.create_dataset("loc_fock", data=mf.loc_fock)
        f.create_dataset("eo", data=mf.eo)




def driver(mklfile, path_mkl2fch, mklfile_loc=None):
    # step 1, load mol, basis and density-fitting
    # step 2, load MOs    os.environ['basis_orca'] = '1'
    # step 3, save HF MOs
    # step 4, save localized MOs
    os.environ['basis'] = 'def2-TZVP'

    mol = get_mole(None)
    mf = make_df(mol)
    load_orca_mo(mf, mklfile, path_mkl2fch)

    save_chkhf(mf)



    if mklfile_loc:
        load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch)
        save_chkloc(mf)





    # save
    

if __name__ == "__main__":
    mklfile = "water_def2tzvp.mkl"
    mklfile_loc = "water_def2tzvp_loc.mkl"
    path_mkl2fch = "/home/nsw/software/mokit/bin/mkl2fch"
    driver(mklfile, path_mkl2fch, mklfile_loc)

    # verification is completed on oxygen01
    # at ~/OSVMP2_ml_gen/work/test/mywater


