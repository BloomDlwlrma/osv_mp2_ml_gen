import numpy as np
import scipy
import ctypes
import time
from pyscf import gto
from pyscf.df import addons
from pyscf.gto.moleintor import make_loc
#from pyscf.gto.moleintor import getints, getints2c, getints3c, getints4c, getints_by_shell
from pyscf.scf import _vhf
from osvmp2.osvutil import *
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm

def read_xyz(xyz_file):
    def get_name(xyz):
        if '/' in xyz:
            rev_count = np.arange(1,len(xyz), dtype='i')*(-1)
            for i in rev_count:
                if xyz[i] == '/': break
            xyz_name = xyz[i+1:-4]
        else:
            xyz_name = xyz[:-4]
        return xyz_name
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        coord = ""
        for l in lines[2:]:
            coord += l
        return coord
def get_slice(rank_list, job_size=None, job_list=None):
    if type(job_list) == type(None):
        job_list = range(job_size)
    else:
        job_size = len(job_list)

    n_rank = len(rank_list)
    n_node = nrank//shm_comm.size
    rank_slice = [range(i, i+nrank//n_node) for i in np.arange(0, nrank, step=nrank//n_node)]
    if n_rank > shm_comm.size:
        rank_list = []
        for rank_i, node_i in itertools.product(range(nrank//n_node), range(n_node)):
            rank_list.append(rank_slice[node_i][rank_i])
    
    n_jobs = [0]*n_rank
    for i in range(job_size):
        rank_idx = rank_list[i%n_rank]
        n_jobs[rank_idx] += 1
    #n_jobs = [no for no in n_jobs if no > 0]
    slice = [None] * n_rank
    idx0 = 0
    for idx, n_j in enumerate(n_jobs):
        if n_j > 0:
            slice[idx] = job_list[idx0:idx0+n_j]
            idx0 += n_j
    return slice

def combine_shells(shell_slice):
    #shell_slice = sorted(shell_slice, key=lambda s: (s[0], s[2]))
    '''int_slice = []
    for idx, si in enumerate(shell_slice):
        if idx == 0:
            slice_i = si
            a0, a1, b0, b1 = si
        else:
            c0, c1, d0, d1 = si
            if (c0 == a0):
                if d0 == b1:
                    slice_i = [a0, a1, slice_i[2], d1]
                else:
                    int_slice.append(slice_i)
                    slice_i = si
            else:
                int_slice.append(slice_i)
                slice_i = si
            a0, a1, b0, b1 = si
        if idx == (len(shell_slice)-1):
            int_slice.append(slice_i)
    shell_slice = []
    for idx, si in enumerate(int_slice):
        if idx == 0:
            slice_i = si
            a0, a1, b0, b1 = si
        else:
            c0, c1, d0, d1 = si
            if (d0 == b0):
                if (c0 == a1) and (b1 == d1):
                    slice_i = [slice_i[0], c1, b0, b1]
                else:
                    shell_slice.append(slice_i)
                    slice_i = si
            else:
                shell_slice.append(slice_i)
                slice_i = si
            a0, a1, b0, b1 = si
        if idx == (len(int_slice)-1):
            shell_slice.append(slice_i)'''
    
    shell_com = []
    b_list = []
    a0_pre = shell_slice[0][0]
    for idx, (a0, a1, b0, b1) in enumerate(shell_slice):
        if a0 != a0_pre:
            shell_com.append(b_list)
            b_list = [[a0, a1, b0, b1]]
        else:
            if b_list == []:
                b_list.append([a0, a1, b0, b1])
            else:
                if b0 == b_list[-1][-1]:
                    b_list[-1][-1] = b1
                else:
                    b_list.append([a0, a1, b0, b1])
        if idx == len(shell_slice)-1:
            shell_com.append(b_list)
        a0_pre = a0
    idx_list = []
    idx_i = []
    for idx, shell_i in enumerate(shell_com):
        b_list = []
        for a0, a1, b0, b1 in shell_i:
            b_list.extend([b0, b1])
        if idx == 0:
            idx_i.append(idx)
        else:
            if b_list == blist_pre:
                idx_i.append(idx)
            else:
                idx_list.append(idx_i)
                idx_i = [idx]
        if idx == len(shell_com)-1:
            idx_list.append(idx_i)
        blist_pre = b_list
    shell_slice = []
    for idx_i in idx_list:
        a0, a1 = shell_com[idx_i[0]][0][0], shell_com[idx_i[-1]][0][1]
        for ax0, ax1, b0, b1 in shell_com[idx_i[0]]:
            shell_slice.append([a0, a1, b0, b1])

    return shell_slice
def collect_slice(shell_seg, naop_shell, max_sum):
    shell_i = []
    len_si = 0
    shell_slice = []
    len_list = []
    for idx, len_i in enumerate(naop_shell):
        shell_i.append(shell_seg[idx])
        len_si += len_i
        idx_next = idx+1
        if idx_next > len(naop_shell)-1:
            idx_next = idx
        if (len_si + naop_shell[idx_next] > max_sum) or idx==(len(naop_shell)-1):
            shell_slice.append(shell_i)
            shell_i = []
            len_list.append(len_si)
            len_si = 0
    return shell_slice, len_list
def OPTPartition(n_slice, shell_seg, naop_shell, match_num=False):#match_num=True):
    #max_sum = get_max_sum(naop_shell, len(naop_shell), n_slice)
    max_sum = sum(naop_shell)//n_slice + 10
    shell_slice, len_list = collect_slice(shell_seg, naop_shell, max_sum)
    len_slice = len(len_list)
    shell_pre = shell_slice
    len_pre = len_list
    step = 0
    var_i = 10
    if (len_slice != n_slice): #and match_num:
        while len_slice != n_slice:
            shell_slice, len_list = collect_slice(shell_seg, naop_shell, max_sum)
            len_slice = len(len_list)
            if len_slice < n_slice:
                if (step > 20) and (len(len_pre)>n_slice):
                    break
                if step > 50:
                    break
                max_sum -= var_i
                shell_pre = shell_slice
                len_pre = len_list
            elif len_slice > n_slice:
                max_sum += var_i
                shell_pre = shell_slice
                len_pre = len_list
            step += 1
            var_i = var_i//2
            if var_i == 0:
                var_i = 1
    shell_slice_com = []
    for shell_i in shell_slice:
        shell_slice_com.append(combine_shells(shell_i))
    return shell_slice_com, len_list

def alloc_shell(shell_slice, naop_shell):
    shell_slice, len_list = OPTPartition(nrank, shell_slice, naop_shell)
    if len(len_list) < nrank:
        shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
        for idx, si in enumerate(shell_slice):
            if si is not None:
                shell_slice[idx] = si[0]
    return shell_slice

def shell_prescreen(mol, auxmol, log, shell_slice=None, shell_tol=1e-10, upper=False, mo_coeff=None, meth_type='RHF'):
    def distance_atoms(coord0, coord1):
        return np.linalg.norm(np.asarray(coord1)-np.asarray(coord0))
    def get_gr(mol):
        '''int_r = np.einsum('ijk->jk', (mol.intor_symmetric('int1e_r'))**2)
        int_rr = np.einsum('ijk->jk', (mol.intor_symmetric('int1e_rr'))**2)
        return np.sqrt(int_r+int_rr)'''

        opt = _vhf.VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond')
        #opt.direct_scf_tol = 1e-13
        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFsetnr_direct_scf')
        mol_q_cond = lib.frompointer(opt._this.contents.q_cond, mol.nbas**2)
        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = np.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        #if mo_coeff is not None:

        return mol_q_cond*np.mean(aux_q_cond)
    nshell = mol.nbas
    nao = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    ao_loc = make_loc(mol._bas, 'sph')
    if shell_slice is None:
        log.info('\nBegin Cauchy-Schwartz prescreening for %s...'%meth_type)
        aoslice = mol.aoslice_by_atom()
        '''atom_coords = mol.atom_coords()
        dist_shell = np.zeros((mol.nbas, mol.nbas))
        for atm0, slice0 in enumerate(aoslice):
            a0, a1 = slice0[:2]
            coord0 = atom_coords[atm0]
            idxlist1 = range(mol.natm)[atm0+1:]
            if len(idxlist1) > 0:
                for atm1 in idxlist1:
                    b0, b1 = aoslice[atm1][:2]
                    coord1 = atom_coords[atm1]
                    d_i = distance_atoms(coord0, coord1)
                    dist_shell[a0:a1, b0:b1] = d_i
                    dist_shell[b0:b1, a0:a1] = d_i'''
        from osvmp2.ga_addons import get_shared
        win_gr, gr = get_shared((nshell, nshell))
        if shm_rank == 0:
            t0 = time_now()
            gr[:] = get_gr(mol).reshape(nshell, nshell)
            t_gr = time_now() - t0
            print_time(['Schwartz integral', t_gr], log=log)
        shm_comm.Barrier()
        ao_atm_offset = mol.offset_nr_by_atom()
        kept_shellp = np.full((nshell, nshell), False, dtype=bool)
        #kept_atoms = np.full((mol.natm, mol.natm), False, dtype=bool)
        atm_list = range(mol.natm)
        shell_list = range(mol.nbas)
        shell_slice = []
        naop_shell = []
        n_total = 0
        n_screened = 0
        s_opt = 1
        t0 = time_now()
        if s_opt == 0:
            atom_list = range(mol.natm)
            for atm0, (a0, a1, al0, al1) in enumerate(ao_atm_offset):
                a_list = range(a0, a1)
                #atm1_list = atom_list[atm0:]
                #for atm1 in atm1_list:
                 #   b0, b1, be0, be1 = ao_atm_offset[atm1]
                for atm1, (b0, b1, be0, be1) in enumerate(ao_atm_offset):
                    b_list = range(b0, b1)
                    if np.mean(gr[a0:a1, b0:b1]) > shell_tol:
                        for ai, bi in itertools.product(a_list, b_list):
                            shell_slice.append([ai, ai+1, bi, bi+1])
                            naop_i = (ao_loc[ai+1]-ao_loc[ai])*(ao_loc[bi+1]-ao_loc[bi])
                            naop_shell.append(naop_i)
                    else:
                        n_screened += (al1-al0)*(be1-be0)
                    n_total += (al1-al0)*(be1-be0)
        else:
            '''for a0, b0 in itertools.product(shell_list, shell_list):
                #gr_list = []
                #for ao0 in range(ao_loc[a0],  ao_loc[a0+1]):
                #    for ao1 in range(ao_loc[b0],  ao_loc[b0+1]):
                #        gr_list.append(gr[ao0, ao1])
                #if (np.mean(gr_list) > shell_tol) or (np.mean(gr_list)==0 and dist_shell[a0, b0]<5):
                #if upper and (a0 > b0): continue
                #if a0 <= b0:
                naop_i = (ao_loc[a0+1]-ao_loc[a0])*(ao_loc[b0+1]-ao_loc[b0])
                if gr[a0, b0] > shell_tol:
                    shell_slice.append([a0, a0+1, b0, b0+1])
                    naop_shell.append(naop_i)
                else:
                    n_screened += naop_i
                n_total += naop_i'''
            for atm0, (a0, a1, al0, al1) in enumerate(ao_atm_offset):
                for atm1, (b0, b1, be0, be1) in enumerate(ao_atm_offset):
                    if (atm0 == atm1) or (np.amax(gr[a0:a1, b0:b1]) > shell_tol):
                        kept_shellp[a0:a1, b0:b1] = True
                    else:
                        n_screened += (al1-al0)*(be1-be0)
                    n_total += (al1-al0)*(be1-be0)
            for a0, b0 in itertools.product(shell_list, shell_list):
                if kept_shellp[a0, b0]:
                    shell_slice.append([a0, a0+1, b0, b0+1])
                    naop_shell.append((ao_loc[a0+1]-ao_loc[a0])*(ao_loc[b0+1]-ao_loc[b0]))
        #####
        shell_slice = alloc_shell(shell_slice, naop_shell)
        t_sel = time_now() - t0
        print_time(['shell pairs selecting', t_sel], log=log)
        shm_comm.Barrier()
        win_gr.Free()
        if irank == 0: 
            msg = "    The threshold for AO pair screening: %.2E"%shell_tol
            msg += "\n    %d out of %d AO pairs are screened, sparsity: %.2f percent"%(n_screened, n_total, 100*float(n_screened)/float(n_total))
            log.info(msg)
        
    return shell_slice



def get_slice_rank(mol, shell_slice, full_pair=False, aslice=False):
    nshell = mol.nbas
    ao_loc = make_loc(mol._bas, 'sph')
    
    if full_pair:
        slice_rank = []
        for si in shell_slice:
            if si is not None:
                slice_rank.extend(si)
        slice_rank = combine_shells(slice_rank)
    elif aslice:
        slice_flat = []
        for si in shell_slice:
            if si is not None:
                slice_flat.extend(si)
        na_list = np.zeros(nshell, dtype='i')
        a_list = [None]*nshell
        for si in slice_flat:
            a0, a1, b0, b1 = si
            ao_B0, ao_B1 = ao_loc[b0], ao_loc[b1]
            nao1 = ao_B1 - ao_B0
            for ai in range(a0, a1):
                nao0 = ao_loc[ai+1] - ao_loc[ai]
                na_list[ai] += nao0*nao1
                slice_i = [ai, ai+1, b0, b1]
                if a_list[ai] is None:
                    a_list[ai] = [slice_i]
                else:
                    a_list[ai].append(slice_i)
        max_sum = get_max_sum(na_list, len(na_list), nrank)
        shell_slice, len_list = collect_slice(a_list, na_list, max_sum)
        len_slice = len(len_list)
        shell_pre = shell_slice
        lenlist_pre = len_list
        
        if (len_slice < nrank):
            ncycle = 0
            while len_slice != nrank:
                shell_slice, len_list = collect_slice(a_list, na_list, max_sum)
                len_slice = len(len_list)
                if len_slice < nrank:
                    max_sum -= 1
                    shell_pre = shell_slice
                    lenlist_pre = len_list
                if (len_slice > nrank) or (max_sum < 0):
                    shell_slice = shell_pre
                    len_list = lenlist_pre
                    break
                ncycle += 1
                if ncycle > 50:
                    break
        
        shell_slice_adjusted = []
        for shell_i in shell_slice:
            slice_i = []
            for si in shell_i:
                slice_i += si
            shell_slice_adjusted.append(slice_i)
        shell_slice_com = []
        for shell_i in shell_slice_adjusted:
            shell_slice_com.append(combine_shells(shell_i))
        if len(len_list) < nrank:
            shell_slice_com = get_slice(rank_list=range(nrank), job_list=shell_slice_com)
            for idx, si in enumerate(shell_slice_com):
                if si is not None:
                    shell_slice_com[idx] = si[0]
        ao_slice = [None]*nrank
        for rank_i, slice_i in enumerate(shell_slice_com):
            if slice_i is not None:
                ao_i = []
                a0_pre, a1_pre = 0, 0
                for si in slice_i:
                    a0, a1 = si[:2]
                    if (a0 != a0_pre) or (a1 != a1_pre):
                        ao_i += [ao_loc[a0], ao_loc[a1]]
                    a0_pre, a1_pre = a0, a1
                ao_slice[rank_i] = [min(ao_i), max(ao_i)]
        slice_rank = shell_slice_com[irank]

    else:
        slice_rank = shell_slice[irank]
    if aslice:
        return ao_slice, slice_rank
    else:
        return slice_rank




