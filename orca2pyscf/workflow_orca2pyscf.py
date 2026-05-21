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
import glob
import h5py
import subprocess
import numpy as np
from pathlib import Path
from pyscf import gto, scf
from pyscf.df import addons
from mokit.lib.fch2py import fch2py
from mokit.lib.ortho import check_orthonormal
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def get_required_env(name):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_recon_tol():
    return float(os.environ.get('ORCA_RECON_TOL', 1e-6))


def get_recon_policy():
    policy = os.environ.get('ORCA_RECON_POLICY', 'STRICT').strip().upper()
    if policy not in {'STRICT', 'SKIP'}:
        raise ValueError(f"Unsupported ORCA_RECON_POLICY={policy}; expected STRICT or SKIP")
    return policy

DEFAULT_AUXBASIS = {
    # AO basis       JK-fit                     MP2-fit
    'ccpvdz'      : ('cc-pvdz-jkfit'          , 'cc-pvdz-ri'         ),
    'ccpvtz'      : ('cc-pvtz-jkfit'          , 'cc-pvtz-ri'         ),
    'augccpvtz'   : ('aug-cc-pvtz-jkfit'      , 'aug-cc-pvtz-ri'     ),
    'def2svp'     : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    'def2tzvp'    : ('def2-tzvp-jkfit'        , 'def2-tzvp-ri'       ),
    '321g'        : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '631g'        : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '631g*'       : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '631g**'      : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '631+g**'     : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
}


def normalize_basis_key(basis):
    return basis.lower().replace('-', '')


def write_chk_metadata(chkfile, mf, chk_kind, source_mkl=None, source_loc_mkl=None):
    with h5py.File(chkfile, 'a') as f:
        meta = f.require_group('meta')
        meta.attrs['chk_kind'] = chk_kind
        meta.attrs['basis_raw'] = str(mf.mol.basis)
        meta.attrs['basis_key'] = normalize_basis_key(str(mf.mol.basis))
        meta.attrs['nao'] = int(mf.mol.nao)
        meta.attrs['no'] = int(np.sum(mf.mo_occ > 0))
        meta.attrs['natom'] = int(mf.mol.natm)
        meta.attrs['atom_signature'] = ','.join([mf.mol.atom_pure_symbol(i) for i in range(mf.mol.natm)])
        meta.attrs['auxbasis_hf'] = str(os.environ.get('auxbasis_hf', ''))
        meta.attrs['auxbasis_mp2'] = str(os.environ.get('auxbasis_mp2', ''))
        if source_mkl is not None:
            meta.attrs['source_mkl'] = str(source_mkl)
        if source_loc_mkl is not None:
            meta.attrs['source_loc_mkl'] = str(source_loc_mkl)

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
    mybasis = [i.lower().replace('-', '') for i in DEFAULT_AUXBASIS.keys()]

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
    basis = normalize_basis_key(basis)
    if basis not in DEFAULT_AUXBASIS:
        raise NotImplementedError(f"Auxiliary basis {basis} not available")

    if mp2fit:
        auxbasis = DEFAULT_AUXBASIS[basis][1]
    else:
        auxbasis = DEFAULT_AUXBASIS[basis][0]

    auxbasis_dic = {}
    atom_list = get_atoms(mol)
    for atm in set(atom_list):
        auxbasis_dic[atm] = auxbasis
    return auxbasis_dic

def get_mole(mol, xyz_file=None):
    if mol == None:
        if xyz_file is None:
             natom, coord = read_xyz(sys.argv[1])
        else:
             natom, coord = read_xyz(xyz_file)

        basis_raw = os.environ.get("basis", 'def2-svp')
        use_ecp = bool(int(os.environ.get("use_ecp", 0)))
        basis_molpro = bool(int(os.environ.get("basis_molpro", 0)))
        basis_orca = bool(int(os.environ.get("basis_orca", 0)))
        verbose = 3
        if basis_molpro:
            pass
        elif basis_orca:
            print("Building molecule with ORCA basis set")
            mole = get_M_orca(coord, basis_raw, verbose)
         
        elif use_ecp or "Be" in coord:
            pass
        else:
            mole = gto.M()
            mole.atom = coord
            mole.basis = basis_raw
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

    subprocess.run([path_mkl2fch, mklfile], check=True)

def load_orca_mo(mf, mklfile, path_mkl2fch):
    mkl2fch(mklfile, path_mkl2fch)
    fchfile = Path(mklfile).with_suffix(".fch")
    nao = mf.mol.nao

    mf.mo_coeff = fch2py(fchfile, nao, nao, 'a')
    mf._source_mkl = str(mklfile)
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

def save_chkhf(mf, filename='hf_mat.chk'):
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
    chkhf = filename
    access_chkfile(chkhf, 'w', [mf.dm, mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.mocc, hfe])
    write_chk_metadata(chkhf, mf, chk_kind='hf', source_mkl=getattr(mf, '_source_mkl', None))


    # dm, mo_energy, mo_coeff, mo_occ, mocc, hfe

def load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch, recon_tol=None, recon_policy=None):
    if recon_tol is None:
        recon_tol = get_recon_tol()
    if recon_policy is None:
        recon_policy = get_recon_policy()

    mkl2fch(mklfile_loc, path_mkl2fch)
    fchfile = Path(mklfile_loc).with_suffix(".fch")
    nao = mf.mol.nao

    mf.o = fch2py(fchfile, nao, nao, 'a')
    mf._source_loc_mkl = str(mklfile_loc)
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

    recon_err = np.max(np.abs(np.dot(mf.mo_coeff[:, mf.mo_occ>0], mf.uo) - mf.o))
    uo_res = recon_err <= recon_tol
    print(f"Checking uo reconstruction with tolerance ({recon_tol:.1e}) under policy {recon_policy}: ", uo_res)
    if not uo_res:
        msg = f"Localized orbital reconstruction failed: max error={recon_err:.3e} > tol={recon_tol:.3e}"
        if recon_policy == 'STRICT':
            raise ValueError(msg)
        print(f"WARNING: {msg}; continuing because ORCA_RECON_POLICY=SKIP")
    mf.loc_fock = np.dot(mf.uo.T, 
      np.dot(
          np.dot(mf.mo_coeff.T, np.dot(mf.fock, mf.mo_coeff))[:no, :no],
        mf.uo))
    mf.eo = np.diag(mf.loc_fock)
    print("Localized orbital `energies`\n", mf.eo)

def save_chkloc(mf, filename='loc_var.chk'):
    chkfile_loc = filename
    with h5py.File(chkfile_loc, 'w') as f:
        f.create_dataset("uo", data=mf.uo)
        f.create_dataset("o", data=mf.o)
        f.create_dataset("loc_fock", data=mf.loc_fock)
        f.create_dataset("eo", data=mf.eo)
    write_chk_metadata(
        chkfile_loc,
        mf,
        chk_kind='loc',
        source_mkl=getattr(mf, '_source_mkl', None),
        source_loc_mkl=getattr(mf, '_source_loc_mkl', None),
    )

def driver(mklfile, path_mkl2fch, mklfile_loc=None, xyz_file=None, output_dir=None, hf_chk_name='hf_mat.chk', loc_chk_name='loc_var.chk'):
    # step 1, load mol, basis and density-fitting
    # step 2, load MOs    os.environ['basis_orca'] = '1'
    # step 3, save HF MOs
    # step 4, save localized MOs
    
    # Use provided output directory if available
    cwd = os.getcwd()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    recon_tol = get_recon_tol()
    recon_policy = get_recon_policy()

    try:
        mol = get_mole(None, xyz_file)
        mf = make_df(mol)
        load_orca_mo(mf, mklfile, path_mkl2fch)

        if mklfile_loc:
            if recon_policy == 'STRICT':
                load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch, recon_tol=recon_tol, recon_policy=recon_policy)
                save_chkhf(mf, hf_chk_name)
                save_chkloc(mf, loc_chk_name)
            else:
                save_chkhf(mf, hf_chk_name)
                # We need absolute path for mklfile_loc if we changed directory, or relative to original cwd
                # Since we pass full paths from main, it should be fine.
                load_orca_loc_mo(mf, mklfile_loc, path_mkl2fch, recon_tol=recon_tol, recon_policy=recon_policy)
                save_chkloc(mf, loc_chk_name)
        else:
            save_chkhf(mf, hf_chk_name)
    finally:
        os.chdir(cwd)
    # save
    

if __name__ == "__main__":
    # --- Infrastructure paths from environment (exported by shell script) ---
    WORK_ROOT = get_required_env("WORK_ROOT")
    OUTPUT_ROOT = get_required_env("OUTPUT_ROOT")
    XYZ_ROOT = get_required_env("XYZ_ROOT")
    MOKIT_BIN = get_required_env("MOKIT_BIN")
    
    # --- Read parameters from environment variables ---
    method = get_required_env("METHOD")
    basis_tag = get_required_env("BASIS_TAG")
    ao_basis = get_required_env("AO_BASIS")
    start_mol = int(get_required_env("START_MOL"))
    end_mol = int(get_required_env("END_MOL"))
    
    # --- Construct paths dynamically from environment variables ---
    mkl_dir = os.path.join(OUTPUT_ROOT, f"orca_mkl_{method}_{basis_tag}")
    loc_mkl_dir = os.path.join(OUTPUT_ROOT, f"orca_locmkl_{method}_{basis_tag}")
    xyz_dir = XYZ_ROOT
    path_mkl2fch = MOKIT_BIN
    output_base_dir = os.path.join(OUTPUT_ROOT, "orca2pyscf", "source_files", f"dsgdb9nsd_{method}_{basis_tag}")
    
    os.environ['basis'] = ao_basis
    # os.environ['basis_orca'] = '1'

    all_files = glob.glob(os.path.join(mkl_dir, "**", "*.mkl"), recursive=True)
    if not all_files:
        print(f"No MKL files found in {mkl_dir}")
        sys.exit(1)
    
    def get_mol_id(filepath):
        basename = os.path.basename(filepath)
        try:
            parts = basename.split('_')
            return int(parts[1])
        except (IndexError, ValueError):
            return float('inf') # Put non-matching at end
    all_files.sort(key=get_mol_id)
    print(f"Found {len(all_files)} total MKL files in source directory.")

    # --- Filtering Logic ---
    target_files = list(all_files)

    # 1. Apply range filter (start_mol, end_mol)
    if start_mol is not None or end_mol is not None:
        range_filtered = []
        for f in target_files:
            mid = get_mol_id(f)
            if start_mol is not None and mid < start_mol:
                continue
            if end_mol is not None and mid > end_mol:
                continue
            range_filtered.append(f)
        target_files = range_filtered
        print(f"After range filtering ({start_mol}-{end_mol}): {len(target_files)} files.")

    # --- Execution Loop ---
    if not target_files:
        print("No files to process after filtering.")
        sys.exit(0)
    print(f"Starting processing of {len(target_files)} molecules...")

    for mkl_path in target_files:
        basename = os.path.basename(mkl_path)
        parts = basename.split('_')
        
        if len(parts) < 4:
            print(f"Skipping {basename}: Unexpected filename format")
            continue
            
        mol_id = f"{parts[0]}_{parts[1]}"
        
        try:
            mol_num = int(parts[1])
            chunk_idx = (mol_num - 1) // 16000
            chunk_dir_name = f"{chunk_idx * 16000 + 1}_{(chunk_idx + 1) * 16000}"
        except ValueError:
            chunk_dir_name = "other"

        # Determine paths
        loc_mkl_name = basename.replace(".mkl", "_loc.mkl") 
        # Locate loc_mkl_path robustly by just looking for the file inside chunk dir
        loc_mkl_path = os.path.join(loc_mkl_dir, chunk_dir_name, loc_mkl_name)
        
        xyz_name = f"{mol_id}.xyz"
        # Using recursive logic to find xyz file in case they are nested
        xyz_search = glob.glob(os.path.join(xyz_dir, "**", xyz_name), recursive=True)
        if not xyz_search:
            print(f"Skipping {mol_id}: could not find {xyz_name} in {xyz_dir}")
            continue
        xyz_path = xyz_search[0]
        
        chunk_dir_path = os.path.join(output_base_dir, chunk_dir_name)
        # We output all structure files in the chunk directory directly
        mol_output_dir = chunk_dir_path
        os.makedirs(mol_output_dir, exist_ok=True)
        
        hf_chk_name = f"hf_mat_{basename.replace('.mkl', '.chk')}"
        loc_chk_name = f"loc_var_{basename.replace('.mkl', '.chk')}"
        
        hf_chk_path = os.path.join(mol_output_dir, hf_chk_name)
        loc_chk_path = os.path.join(mol_output_dir, loc_chk_name)
        
        if os.path.exists(hf_chk_path) and os.path.exists(loc_chk_path):
            # print(f"Skipping {mol_id}: target files {hf_chk_name} and {loc_chk_name} already exist")
            continue
        
        try:
            driver(mkl_path, path_mkl2fch, loc_mkl_path, xyz_file=xyz_path, output_dir=mol_output_dir, hf_chk_name=hf_chk_name, loc_chk_name=loc_chk_name)
            print(f"Successfully processed {mol_id}")
        except Exception as e:
            print(f"Error processing {mol_id}: {e}")
            import traceback
            traceback.print_exc()

    # verification is completed on oxygen01
    # at ~/OSVMP2_ml_gen/work/test/mywater


