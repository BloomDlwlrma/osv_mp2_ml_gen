# To reuse legacy OSVMP2 IO of building molecules, which
# 1. Gets coordinates from xyz files
# 2. Set basis sets and other parameters
# 3. Loads HF and localized orbitals from chkfiles

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
import json
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

ALL_BASIS_SETS = [
    "cc-pVDZ", "aug-cc-pVDZ", "cc-pVTZ", "aug-cc-pVTZ", "cc-pVQZ", "aug-cc-pVQZ", "cc-pV5Z", "aug-cc-pV5Z",
    "def2-SVP", "def2-SVPD", "def2-TZVP", "def2-mTZVP", "def2-TZVPD", "def2-TZVPP", "def2-mTZVPP", "def2-TZVPPD",
    "def2-QZVP", "def2-QZVPP", "def2-QZVPPD", 
    "STO-3G", "3-21G", "6-31G", "6-31+G", "6-31++G", "6-311G", "6-311+G", "6-311++G"
]

ALL_HF_AUXBASIS = [
    "cc-pVDZ-jkfit", "aug-cc-pVDZ-jkfit", "cc-pVTZ-jkfit", "aug-cc-pVTZ-jkfit", "cc-pVQZ-jkfit", "aug-cc-pVQZ-jkfit", "cc-pV5Z-jkfit", "aug-cc-pV5Z-jkfit",
    "def2-SVP-jkfit", "def2-SVPD-jkfit", "def2-TZVP-jkfit", "def2-mTZVP-jkfit", "def2-TZVPD-jkfit", "def2-TZVPP-jkfit", "def2-mTZVPP-jkfit", "def2-TZVPPD-jkfit",
    "def2-QZVP-jkfit", "def2-QZVPP-jkfit", "def2-QZVPPD-jkfit"
]

ALL_MP2_AUXBASIS = [
    "cc-pVDZ-ri", "aug-cc-pVDZ-ri", "cc-pVTZ-ri", "aug-cc-pVTZ-ri", "cc-pVQZ-ri", "aug-cc-pVQZ-ri", "cc-pV5Z-ri", "aug-cc-pV5Z-ri",
    "def2-SVP-ri", "def2-SVPD-ri", "def2-TZVP-ri", "def2-mTZVP-ri", "def2-TZVPD-ri", "def2-TZVPP-ri", "def2-mTZVPP-ri", "def2-TZVPPD-ri",
    "def2-QZVP-ri", "def2-QZVPP-ri", "def2-QZVPPD-ri"
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert ORCA MKL files to PySCF chkfiles")
    parser.add_argument("--mol", "-M", type=str, default=None, help="Molecule name to process")
    parser.add_argument("--basis", type=str, default=None, help="Basis set to process (e.g. cc-pVTZ)")
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
    '''
    Docstring for get_M_orca
    Function: get_M_orca builds a PySCF molecule object using ORCA basis sets.
    Inputs:
        coord: str, coordinates of the molecule in XYZ format
        basis: str, basis set name (e.g., "cc-pVTZ", "def2-TZVP")
        verbose: int, verbosity level for PySCF molecule building
    Outputs:
        mole: PySCF molecule object with specified basis set
    '''
    
    basis_mode = os.environ.get("BASIS_MODE", "verify")
    
    if basis_mode == "verify":
        # Load from json
        json_path = os.path.join(os.path.dirname(__file__), "mybasis_verify.json")
        try:
            with open(json_path, 'r') as f:
                mybasis = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load mybasis_verify.json: {e}. Using default fallback.")
            mybasis = ["cc-pVTZ", "def2-TZVP"]
    elif basis_mode == "all":
        mybasis = ALL_BASIS_SETS
    else: # specific
        mybasis = [basis]

    mybasis_norm = [i.lower().replace('-', '') for i in mybasis]

    if not basis.lower().replace('-', '') in mybasis_norm:
        if basis_mode == "verify":
             logging.warning(f"ORCA basis set {basis} not in verified list. Proceeding anyway as it might be intended, but check configuration.")
        else:
             logging.info(f"Using specific basis: {basis}")

    mole = gto.Mole()
    mole.atom = coord
    mole.basis = basis
    mole.charge = int(os.environ.get("charge", 0))
    mole.spin = int(os.environ.get("spin", 0))
    mole.build(verbose=verbose)

    return mole

def get_aux_orca(mol, basis, mp2fit=False):  
    # Common logic for basis set handling
    basis = basis.lower().replace('-', '')
    if mp2fit:
        fit_type = "ri"
    else:
        fit_type = "jkfit"
    
    auxbasis = basis + fit_type
    
    basis_mode = os.environ.get("BASIS_MODE", "verify")

    if basis_mode == "specific":
        logging.info("Using specific basis set mode, bypassing auxiliary basis validation")
        # No validation for specific mode
    else:
        if basis_mode == "verify":
            # logging.info("Verifying ORCA auxiliary basis sets")
            myhf_auxbasis = ["cc-pVTZ-jkfit", "def2-TZVP-jkfit"]
            mymp2_auxbasis = ["cc-pVTZ-ri", "def2-TZVP-ri"]
        elif basis_mode == "all":
            # logging.info("Testing ALL ORCA auxiliary basis sets")
            myhf_auxbasis = [i.lower().replace('-', '') for i in ALL_HF_AUXBASIS]
            mymp2_auxbasis = [i.lower().replace('-', '') for i in ALL_MP2_AUXBASIS]
        else:
             logging.error(f"Mode {basis_mode} not recognized")
        if auxbasis not in [ibas.lower().replace('-', '') for ibas in myhf_auxbasis+mymp2_auxbasis]:
            raise NotImplementedError(f"Auxiliary basis {auxbasis} not available")

    # Common dictionary construction
    auxbasis_dic = {}
    atom_list = get_atoms(mol)
    for atm in set(atom_list):
        auxbasis_dic[atm] = auxbasis
    return auxbasis_dic

def get_mole(mol, xyz_file=None):
    if mol == None:
        if xyz_file:
             target_file = xyz_file
        elif len(sys.argv) > 1:
             target_file = sys.argv[1]
        else:
             raise ValueError("No XYZ file provided")
             
        natom, coord = read_xyz(target_file)
        basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
        use_ecp = bool(int(os.environ.get("use_ecp", 0)))
        # basis_molpro = bool(int(os.environ.get("basis_molpro", 0)))
        basis_orca = bool(int(os.environ.get("basis_orca", 0)))
        verbose = 3
        # if basis_molpro:
        #     pass
        if basis_orca:
            logging.info("Building molecule with ORCA basis set")
            mole = get_M_orca(coord, basis, verbose, basis)
         
        elif use_ecp or "Be" in coord:
            logging.warning("This workflow is not building molecule with ECP basis set from PySCF")
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

# mf
def make_df(mol, mp2fit):
    """
    Create a density-fitting RHF object
    nao: number of atomic orbitals, generated based on basis function (molecule structure and basis set)
    Outputs:
        mf: a PySCF RHF (restricted Hartree-Fock) calculation object with Density Fitting enabled.
    """
    auxbasis_hf = get_aux_orca(mol, mol.basis, mp2fit=False)
    auxbasis_mp2 = get_aux_orca(mol, mol.basis, mp2fit=True)

    auxmol_hf = addons.make_auxmol(mol, auxbasis_hf)
    auxmol_mp2 = addons.make_auxmol(mol, auxbasis_mp2)
    auxbasis_hf = get_aux_orca(mol, mol.basis, mp2fit=False)
    auxbasis_mp2 = get_aux_orca(mol, mol.basis, mp2fit=True)
    logging.info(f"MP2 density fitting auxiliarty basis: {auxbasis_mp2}")
    logging.info(f"Number of HF auxiliary basis set: {auxmol_hf.nao_nr()}")
    logging.info(f"Number of MP2 auxiliary basis set: {auxmol_mp2.nao_nr()}")

    mf = scf.RHF(mol).density_fit()
    
    # Use MP2 auxiliary basis if requested via args.mp2fit
    if mp2fit == "mp2":
        logging.info("Using MP2 auxiliary basis for SCF density fitting")
        mf.with_df.auxbasis = auxbasis_mp2
        mf.with_df.auxmol = auxmol_mp2
    else:
        logging.info("Using HF auxiliary basis for SCF density fitting")
        mf.with_df.auxbasis = auxbasis_hf
        mf.with_df.auxmol = auxmol_hf
        
    return mf

def check_mkl2fch(mklfile, path_mkl2fch):
    '''
    check: whether mkl2fch binary and mklfile are valid
    mklfile: path of ORCA generated MKL file
        use `orca_2mkl basename -mkl` to convert basename.gbw to MKL file
    path_mkl2fch: path of executable binary `mkl2fch` provided by MOKIT 
    '''
    if not Path(mklfile).is_file():
        raise ValueError(f"MKL file {mklfile} not a valid file")
    if not Path(path_mkl2fch).is_file():
        raise ValueError(f"mkl2fchk binary {path_mkl2fch} not valid")

    subprocess.run([path_mkl2fch, mklfile],check=True)

def load_orca_mo(mf, mklfile, path_mkl2fch):
    '''
    Docstring for load_orca_mo
    Function: load_orca_mo loads molecular orbitals from an ORCA MKL file into a PySCF mean-field object.
    Workflow:
        1. Convert the ORCA MKL file to a formatted checkpoint (.fch) file using the mkl2fch utility.
        2. Read the molecular orbitals from the .fch file and assign them to the mean-field object.
        3. Check the orthonormality of the loaded molecular orbitals.
        4. Set the occupation numbers for a closed-shell system.
        5. Compute the density matrix, Fock matrix, and SCF energy.
    Inputs:
        mf: PySCF mean-field object (e.g., RHF)
        mklfile: str, path to the ORCA MKL file
        path_mkl2fch: str, path to the mkl2fch utility
    Outputs:
        None (the mean-field object is modified in place)
    '''
    check_mkl2fch(mklfile, path_mkl2fch)
    fchfile = Path(mklfile).with_suffix(".fch") 
    nao = mf.mol.nao
    # for ORCA, alpha and beta MOs are the same for closed-shell systems
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
                        array_dic[key_i][:] = scf.hf.make_rdm1(array_dic["mo_coeff"], array_dic["mo_occ"])

    if mode == 'r':
        return arrays

def save_chkhf(mf):
    '''
    Function: save_chkhf saves the **Canonical Hartree-Fock (HF)** molecular orbital data to a checkpoint file. 
    Workflow:
        1. Calculate molecular orbital energies by transforming the Fock matrix into the MO basis.
        2. Determine the occupation numbers for the molecular orbitals.
        3. Compute the occupied molecular orbital coefficients.
        4. Print the molecular orbital energies and occupation numbers.
        5. Save the density matrix, molecular orbital energies, coefficients, occupation numbers, occupied
              molecular orbital coefficients, and SCF energy to a checkpoint file.
    Inputs:
        mf: PySCF mean-field object (e.g., RHF) containing the necessary
    Outputs:
        None (data is saved to hf_mat.chk)
    '''
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
    '''
    Function: load_orca_loc_mo loads localized molecular orbitals from an ORCA MKL file into a PySCF mean-field object.
    Workflow:
        1. Convert the ORCA MKL file to a formatted checkpoint (.fch) file using the mkl2fch utility.
        2. Read the localized molecular orbitals from the .fch file and assign them to the mean-field object.
        3. Check the orthonormality of the loaded localized molecular orbitals.
        4. Set the occupation numbers for a closed-shell system.
        5. Compute the density matrix, Fock matrix, and SCF energy.
        6. Compute the localized Fock matrix in the MO space and its diagonal elements (localized orbital energies).
    Inputs:
        mf: PySCF mean-field object (e.g., RHF)
        mklfile_loc: str, path to the ORCA MKL file containing localized orbitals
        path_mkl2fch: str, path to the mkl2fch utility
    Outputs:
        None (the mean-field object is modified in place)
    '''
    check_mkl2fch(mklfile_loc, path_mkl2fch)
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

def driver(mklfile, path_mkl2fch, mklfile_loc=None, basis_name=None, xyz_file=None):
    # step 1, load mol, basis and density-fitting
    # step 2, load MOs    os.environ['basis_orca'] = '1'
    # step 3, save HF MOs
    # step 4, save localized MOs

    if basis_name:
        os.environ['basis'] = basis_name
    elif 'basis' not in os.environ:
         os.environ['basis'] = 'def2-TZVP'

    mol = get_mole(None, xyz_file=xyz_file)
    mf = make_df(mol, None)
    load_orca_mo(mf, mklfile, path_mkl2fch)

    save_chkhf(mf)

    if mklfile_loc:
        load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch)
        save_chkloc(mf)
    # save
    

if __name__ == "__main__":
    args = parse_arguments()
    path_mkl2fch = "/home/ubuntu/packages/mokit/bin/mkl2fch"
    
    # Define directories
    work_dir = "/home/ubuntu/Shiwei/gen_feature/orca2pyscf"
    xyz_dir = os.path.join(work_dir, "xyz_files")
    source_dir = os.path.join(work_dir, "source_files")
    
    if not os.path.exists(xyz_dir):
        logging.error(f"XYZ directory not found: {xyz_dir}")
        sys.exit(1)

    target_basis_list = []
    
    if args.basis is None:
        os.environ["BASIS_MODE"] = "verify"
        logging.info("Mode: VERIFY (default)")
        json_path = os.path.join(os.path.dirname(__file__), "mybasis_verify.json")
        try:
             with open(json_path, 'r') as f:
                target_basis_list = json.load(f)
        except Exception as e:
                logging.warning(f"Could not load mybasis_verify.json: {e}. Using default fallback.")
                target_basis_list = ["cc-pVTZ", "def2-TZVP"]
                
    elif args.basis.lower() == "all":
        os.environ["BASIS_MODE"] = "all"
        logging.info("Mode: ALL")
        target_basis_list = ALL_BASIS_SETS
        
    else:
        os.environ["BASIS_MODE"] = "specific"
        logging.info(f"Mode: SPECIFIC ({args.basis})")
        target_basis_list = [args.basis]
        
    # Normalize the target list for comparison
    target_basis_norm = [b.lower().replace('-', '') for b in target_basis_list]

    mol_list = []
    if args.mol:
        mol_list = [args.mol]
    else:
        # List all xyz files
        for f in os.listdir(xyz_dir):
            if f.endswith('.xyz'):
                mol_list.append(os.path.splitext(f)[0])

    for mol_name in mol_list:
        xyz_file = os.path.join(xyz_dir, f"{mol_name}.xyz")
        if not os.path.exists(xyz_file):
            if args.mol:
                 logging.error(f"XYZ file not found for {mol_name}: {xyz_file}")
                 sys.exit(1)
            else:
                 logging.warning(f"XYZ file not found for {mol_name}: {xyz_file}")
                 continue
        
        mol_source_dir = os.path.join(source_dir, mol_name)
        if not os.path.exists(mol_source_dir):
             if args.mol:
                 logging.error(f"Source directory not found for {mol_name}: {mol_source_dir}")
                 sys.exit(1)
             else:
                 logging.warning(f"Source directory not found for {mol_name}: {mol_source_dir}")
                 continue

        # Iterate over subdirectories in mol_source_dir to find basis sets
        for item in os.listdir(mol_source_dir):
            item_path = os.path.join(mol_source_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            # Pattern: {mol_name}_{basis}
            # Check if it starts with mol_name and an underscore
            if not item.startswith(mol_name + "_"):
                continue
            
            basis_part = item[len(mol_name)+1:] # extract suffix
            
            # Check if directory contains files (is not empty)
            if not os.listdir(item_path):
                # logging.info(f"Skipping empty directory: {item_path}")
                continue

            if basis_part.lower().replace('-', '') not in target_basis_norm:
                continue
            
            # Files inside
            mkl_name = f"{item}.mkl"
            mkl_loc_name = f"{item}_loc.mkl"
            mklfile = os.path.join(item_path, mkl_name)
            mklfile_loc = os.path.join(item_path, mkl_loc_name)
            
            if os.path.exists(mklfile):
                loc_file = mklfile_loc if os.path.exists(mklfile_loc) else None
                try:
                    logging.info(f"Processing {mol_name} with basis {basis_part} from {item}")
                    driver(mklfile, path_mkl2fch, loc_file, basis_name=basis_part, xyz_file=xyz_file)
                except Exception as e:
                    logging.error(f"Failed processing {mol_name} {basis_part}: {e}")
            else:
                 logging.info(f"MKL file not found: {mklfile}")
