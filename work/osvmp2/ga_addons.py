from genericpath import isfile
import time
import os
import copy
import shutil
from decimal import Decimal
import numbers
import subprocess
import h5py
import scipy
import numpy as np
from pyscf.gto.moleintor import make_loc
from osvmp2 import int_prescreen
from osvmp2.osvutil import *
import mpi4py
mpi4py.rc.thread_level = 'single'
from mpi4py import MPI


#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()   # Size of communicator
irank = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm

win_list = []
n_win = 0

#def get_shared(dim1, dim2=None, dim3=None, dtype='f8'):
def get_shared(shape, dtype='f8'):
    if dtype == 'f8':
        itemsize = MPI.DOUBLE.Get_size()
    elif dtype == 'i':
        itemsize = MPI.INT.Get_size()
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    if irank_shm == 0:
        nbytes = np.prod(shape) * itemsize
    else:
        nbytes = 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm_shm)
    buf, itemsize = win.Shared_query(MPI.PROC_NULL)
    buf = np.array(buf, dtype='B', copy=False)
    shared_obj = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    if irank_shm == 0:
        shared_obj[:] = 0
    comm_shm.Barrier()
    global n_win
    n_win += 1
    win_list.append(win)
    return win, shared_obj

def create_win(var, comm=comm):
    win = MPI.Win.Create(var, comm=comm)
    global n_win
    n_win += 1
    win_list.append(win)
    return win

def free_win(win):
    win.Free()
    global n_win
    n_win -= 1

def fence_and_free(win):
    win.Fence()
    free_win(win)

def free_all_win():
    global n_win
    global win_list
    for win in win_list:
        try:
            win.Free()
            n_win -= 1
        except mpi4py.MPI.Exception:
            pass
    win_list = []

def get_win_col(var):
    if irank_shm == 0:
        win_col = create_win(var, comm=comm)
    else:
        win_col = create_win(None, comm=comm)
    win_col.Fence()
    return win_col

def Accumulate_GA(win=None, var=None, target_rank=0, cross_node=True):
    def acc_ga():
        if cross_node:
            if (irank_shm == 0) and (irank != 0):
                win.Lock(target_rank)
                win.Accumulate(var, target_rank=target_rank, op=MPI.SUM)
                win.Unlock(target_rank)
        elif irank != target_rank:
            win.Lock(target_rank)
            win.Accumulate(var, target_rank=target_rank, op=MPI.SUM)
            win.Unlock(target_rank)
    if win is None:
        win = get_win_col(var)
        acc_ga()
        fence_and_free(win)
    else:
        acc_ga()
        

def Accumulate_GA_shm(win, var_node, var):
    win.Lock(0)
    var_node += var
    win.Unlock(0)

def Get_GA(win, var, target_rank=0, target=None):
    win.Lock(target_rank, lock_type=MPI.LOCK_SHARED)
    if target is None:
        win.Get(var, target_rank=target_rank)
    else:
        win.Get(var, target_rank=target_rank, target=target)
    win.Unlock(target_rank)


def Acc_and_get_GA(var):
    win_col = get_win_col(var)
    if irank_shm == 0 and irank != 0:
        Accumulate_GA(win_col, var, target_rank=0)
    win_col.Fence()
    if irank_shm == 0 and irank != 0:
        Get_GA(win_col, var, target_rank=0)
    fence_and_free(win_col)

def bcast_GA(var, win_col=None):
    if win_col == None:
        if irank_shm == 0:
            win_col = create_win(var, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        win_col.Fence()
        new_win = True
    else:
        new_win = False
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0, lock_type=MPI.LOCK_SHARED)
        win_col.Get(var, target_rank=0)
        win_col.Unlock(0)
    if new_win:
        win_col.Fence()
        free_win(win_col)
    return var

def collect_GA(var, win_col=None):
    if win_col == None:
        if irank_shm == 0:
            win_col = create_win(var, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        win_col.Fence()
        new_win = True
    else:
        new_win = False
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0)
        win_col.Accumulate(var, target_rank=0, op=MPI.SUM)
        win_col.Unlock(0)
    if new_win:
        win_col.Fence()
        free_win(win_col)
    return var

def get_GA_slice(addr, slice_i):
    addr_slice = []
    for i in slice_i:
        addr_slice.append(addr[i])
    
    rank_all, idx_all = zip(*addr_slice)
    rank_list = []
    seg_list = []
    idx_list = []
    rank_pre = -1
    i0_pre, i1_pre = -1, -1
    idx0_pre, idx1_pre = -1, -1
    for idx, rank_i in enumerate(rank_all):
        i0 = slice_i[idx]
        i1 = i0 + 1
        idx0, idx1 = idx_all[idx]
        if (rank_i == rank_pre) and (idx0 == idx1_pre):
            i1_pre = i1
            idx1_pre = idx1
        else:
            if idx != 0:
                rank_list.append(rank_pre)
                seg_list.append([i0_pre, i1_pre])
                idx_list.append([idx0_pre, idx1_pre])
            rank_pre = rank_i
            i0_pre, i1_pre = i0, i1
            idx0_pre, idx1_pre = idx0, idx1
            
        if idx == (len(rank_all)-1):
            rank_list.append(rank_pre)
            seg_list.append([i0_pre, i1_pre])
            idx_list.append([idx0_pre, idx1_pre])
    
    return rank_list, seg_list, idx_list
def get_coords(mol):
    atm_list = []
    for atm in range(mol.natm):
        atm_list.append(mol.atom_pure_symbol(atm))
    xyz_list = mol.atom_coords()
    a = 0
    xyz = []
    for i in xyz_list:
       xyz_for = []
       for j in i.tolist():
           j = format(j, '.9f')
           xyz_for.append(j)
       xyz.append(xyz_for)
       a += 1
    for i in range(len(atm_list)):
        xyz[i].insert(0,atm_list[i])

    lens = []
    for column in zip(*xyz):
        lens.append(max([len(x) for x in column]))
    xyz_for = "  ".join(["{:<" + str(x) + "}" for x in lens])
    for row_i in xyz:
        print(xyz_for.format(*row_i))
    return xyz
#def get_GA_slice_imba(addr, slice_i):
def read_file(f_name, obj_name, idx0=None, idx1=None, buffer=None):
    var = None
    read_sucess = False; count = 0
    while (read_sucess is False) and (count<10):
        try:
            with h5py.File(f_name, 'r') as f:
                if idx0 is None:
                    if buffer is None:
                        var = f[obj_name][:]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[:])
                elif idx1 is None:
                    if buffer is None:
                        var = f[obj_name][idx0]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[idx0])
                else:
                    if buffer is None:
                        var = f[obj_name][idx0:idx1]
                    else:
                        f[obj_name].read_direct(buffer, np.s_[idx0:idx1])
                read_sucess = True
        except IOError as e:
            einfo = e
            read_sucess = False
        count += 1
    if read_sucess is False:
        raise IOError(einfo)
    if buffer is None:
        return var
    
def read_GA(addr, slice_i, buffer, win, dtype='f8', list_col=None, dim_list=None, sup_dim=1, buf_idx_start=0):
    if dtype == 'f8':
        size_unit = 8
    elif dtype == 'i':
        size_unit = 4
    sup_shape = sup_dim
    sup_dim = np.product(sup_dim)
    slice_i = list(sorted(set(slice_i)))
    buf_idx0 = buf_idx_start
    slice_kept = []
    if type(list_col) != type(None):
        for i in slice_i:
            if type(list_col[i]) == type(None):
                break
            slice_kept.append(i)
            slice_i.remove(i)
        for i in slice_kept:
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.product(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
            buf_idx0 = buf_idx1
    if slice_i != []:
        rank_list, seg_list, idx_list = get_GA_slice(addr, slice_i)
        for idx, rank_i in enumerate(rank_list):
            recv_idx0, recv_idx1 = idx_list[idx]
            buf_idx1 = buf_idx0 + (recv_idx1-recv_idx0)
            win.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
            win.Get(buffer[buf_idx0: buf_idx1], target_rank=rank_i, target=[recv_idx0*sup_dim*size_unit, (recv_idx1-recv_idx0)*sup_dim, MPI.DOUBLE])
            win.Unlock(rank_i)
            buf_idx0 = buf_idx1
    if type(list_col) != type(None):
        buf_idx0 = buf_idx_start
        for i in slice_kept:
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.product(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
            buf_idx0 = buf_idx1
        for idx, i in enumerate(slice_i):
            if sup_dim == 1:
                buf_idx1 = buf_idx0 + np.product(dim_list[i])
                list_col[i] = buffer[buf_idx0: buf_idx1].reshape(dim_list[i])
            else:
                buf_idx1 = buf_idx0 + 1
                list_col[i] = buffer[buf_idx0: buf_idx1].reshape(sup_shape)
            buf_idx0 = buf_idx1
        return buffer, list_col
    else:
        return buffer

def get_buff_size(dim_list, slice_i):
    dim_sum = 0
    for i in slice_i:
        dim_sum += np.product(dim_list[i])
    return dim_sum

def get_GA_node(self, slice_list, win_ga, addr_ga, dim_list, len_addr, sup_dim=1):
    if sup_dim == 1:
        buff_size = get_buff_size(dim_list, slice_list)
        win_buff, buff = get_shared(buff_size, dtype='f8')
    else:
        dim0 = len(slice_list)
        if type(sup_dim) == int:
            win_buff, buff = get_shared((dim0, sup_dim), dtype='f8')
        else:
            dim1, dim2 = sup_dim
            win_buff, buff = get_shared((dim0, dim1, dim2), dtype='f8')

    address = [None]*len_addr
    idx0 = 0
    for i in slice_list:
        idx1 = idx0 + np.product(dim_list[i])
        address[i] = [idx0, idx1]
        idx0 = idx1
    
    slice_i = get_slice(rank_list=self.shm_ranklist, job_list=slice_list)[irank_shm]
    if slice_i is not None: 
        buff = read_GA(addr_ga, slice_i, buff, win_ga, dtype='f8', dim_list=dim_list, sup_dim=sup_dim, buf_idx_start=address[slice_i[0]][0])
    comm_shm.Barrier()
    return address, win_buff, buff

def read_GA_node(slice_i, addr_node, buf_node, dim_list, tmp_list=None, buff=None):
    if tmp_list is None:
        i = slice_i[0]
        recv_idx0, recv_idx1 = addr_node[i]
        if buff is not None:
            buff[:] = buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
            return buff
        else:
            return buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
    else:
        tmp_list = [None]*len(tmp_list)
        buf_idx0 = 0
        for i in slice_i:
            buf_idx1 = buf_idx0 + np.product(dim_list[i])
            recv_idx0, recv_idx1 = addr_node[i]
            if buff is None:
                tmp_list[i] = buf_node[recv_idx0: recv_idx1].reshape(dim_list[i])
            else:
                buff[buf_idx0: buf_idx1] = buf_node[recv_idx0: recv_idx1]
                tmp_list[i] = buff[buf_idx0: buf_idx1].reshape(dim_list[i])
            buf_idx0 = buf_idx1
        return tmp_list



'''def get_pairslice(self):
    pair_slice_remote = get_slice(job_list=self.pairlist_remote, rank_list=range(nrank))
    pair_slice_close = get_slice(job_list=self.pairlist_close, rank_list=range(nrank))
    pair_slice = []
    for rank_i in range(nrank):
        if (pair_slice_remote[rank_i] == None) and (pair_slice_close[rank_i] == None):
            pair_slice.append(None)
        else:
            slice_i = []
            if pair_slice_remote[rank_i] is not None:
                slice_i.extend(pair_slice_remote[rank_i])
            if pair_slice_close[rank_i] is not None:
                slice_i.extend(pair_slice_close[rank_i])
            pair_slice.append(sorted(slice_i))
    return pair_slice'''

'''def get_nfit(auxmol, pairlist, atom_close, naux_close):
    from loc_addons import joint_fit_domains_by_atom
    nocc = len(atom_close)
    win_nfit, nfit = get_shared(nocc**2, dtype='i')
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            i = ipair//nocc
            j = ipair%nocc
            if i == j:
                nfit[ipair] = naux_close[i]
            else:
                nfit[ipair] = joint_fit_domains_by_atom(auxmol, [i, j], atom_close, joint_type='union')[1]
    comm.Barrier()
    Acc_and_get_GA(nfit)
    return win_nfit, nfit'''

def get_nfit(auxmol, pairlist, fit_list, nfit_close):
    from osvmp2.loc.loc_addons import joint_fit_domains_by_aux
    nocc = len(fit_list)
    win_nfit, nfit = get_shared(nocc**2, dtype='i')
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            i = ipair//nocc
            j = ipair%nocc
            if i == j:
                nfit[ipair] = nfit_close[i]
            else:
                nfit[ipair] = joint_fit_domains_by_aux(auxmol, [i, j], fit_list, joint_type='union')[1]
    comm.Barrier()
    Acc_and_get_GA(nfit)
    return win_nfit, nfit        

def get_pairslice(self, if_remote=True, if_full=False, pairlist=None, even_adjust=False, log=None):
    def sort_pairlist(self, pairlist):
        if_remote = self.if_remote[pairlist[0]]
        nosv_list = []
        for ipair in pairlist:
            i = ipair // self.no
            j = ipair % self.no
            if if_full:
                nosv_list.append(self.nosv[j])
            else:
                #nosv_list.append(self.nosv[i]*self.nosv[j])
                nosv_list.append([self.nosv[i], self.nosv[j]])

        if self.loc_fit:
            win_nfit, nfit = get_nfit(self.with_df.auxmol, pairlist, self.fit_list, self.nfit)
            log = lib.logger.Logger(self.stdout, self.verbose)
            if if_remote:
                log.info('    Average scr pair nfit: %d'%(np.sum(nfit)/len(pairlist)))
            else:
                log.info('    Average rem pair nfit: %d'%(np.sum(nfit)/len(pairlist)))
        else:
            nfit = [self.naoaux]*self.no**2

        cost_list = []
        for idx, ipair in enumerate(pairlist):
            nosv_i, nosv_j = nosv_list[idx]
            nfit_ij = nfit[ipair]
            if if_remote:
                cost_list.append(nosv_i*nfit_ij*nosv_j)
            else:
                nosv_ij = nosv_i + nosv_j
                #cost_list.append(nosv_ij*nfit_ij*(self.nao+nosv_ij)+nfit_ij**3+2*nfit_ij**2)
                cost_virco = self.nao*self.nv*nosv_ij
                cost_trans = nosv_ij*self.nao*nfit_ij
                if (ipair//self.no) != (ipair%self.no):
                    cost_trans *= 2
                cost_trans += nosv_ij*nfit_ij*nosv_ij
                cost_list.append(cost_virco+cost_trans)
        if self.loc_fit:
            comm_shm.Barrier()
            free_win(win_nfit); nfit=None
        return [ipair for nosv, ipair in sorted(zip(cost_list, pairlist), reverse=True)]
    
    def get_si(cluslist):
        job_size = len(cluslist)
        rank_list = []
        for rank_i, node_i in itertools.product(range(nrank//self.nnode), range(self.nnode)):
            rank_list.append(self.rank_slice[node_i][rank_i])
        job_slice = [None]*nrank
        for idx, clus_i in enumerate(cluslist):
            rank_idx = rank_list[idx%nrank]
            if job_slice[rank_idx] == None:
                job_slice[rank_idx] = [clus_i]
            else:
                job_slice[rank_idx].append(clus_i)
        return job_slice

    def merge_pairslice(pair_slice_remote, pair_slice_close):
        pair_slice = []
        for rank_i in range(nrank):
            if (pair_slice_remote[rank_i] == None) and (pair_slice_close[rank_i] == None):
                pair_slice.append(None)
            else:
                slice_i = []
                if pair_slice_remote[rank_i] is not None:
                    slice_i.extend(pair_slice_remote[rank_i])
                if pair_slice_close[rank_i] is not None:
                    slice_i.extend(pair_slice_close[rank_i])
                pair_slice.append(sorted(slice_i))
        return pair_slice

    if even_adjust:
        if pairlist is None:
            if if_full:
                pairlist_full = sort_pairlist(self, self.pairlist_full)
                pair_slice = get_si(pairlist_full)
            else:
                pairlist_close = sort_pairlist(self, self.pairlist_close)
                if (if_remote) and (self.pairlist_remote != []):
                    pairlist_remote = sort_pairlist(self, self.pairlist_remote)
                    pair_slice = get_si(pairlist_close + pairlist_remote)
                else:
                    pair_slice = get_si(pairlist_close)
        else:
            pairlist = sort_pairlist(self, pairlist)
            pair_slice = get_si(pairlist)
    else:
        if pairlist is None:
            if if_full:
                pair_slice = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_full)
            else:
                pair_slice = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_close)
                if if_remote:
                    pair_slice_remote = get_slice(rank_list=range(nrank), job_list=self.refer_pairlist_remote)
                    pair_slice = merge_pairslice(pair_slice_remote, pair_slice)
        else:
            pair_slice = get_slice(rank_list=range(nrank), job_list=pairlist)
    return pair_slice


#def parallel_feri_GA(self, log):
def parallel_feri_GA(self, df_obj, meth_type, log):
    def gen_feri(index, feri_ga, feri_buffer, low_node):
        s0, s1, p0, p1 = index
        s_slice = (s0, s1, 0, self.mol.nbas, self.mol.nbas, self.mol.nbas+auxmol.nbas)
        feri_tmp = aux_e2(self.mol, auxmol, intor='int3c2e_sph', aosym='s2ij', comp=1, shls_slice=s_slice, out=feri_buffer).T
        feri_tmp = scipy.linalg.solve_triangular(low_node, feri_tmp, lower=True, overwrite_b=True, 
                                                 check_finite=False)
        #ddot(low_node, feri_tmp, out=feri_tmp)
        feri_ga[:, p0:p1] = feri_tmp
        feri_tmp = None
        return feri_ga
    def get_eri_slice(ranklist, rank_i, shell_seg, naop_seg):
        n_rank = len(ranklist)
        if len(shell_seg) < n_rank:
            shell_slice = get_slice(rank_list=ranklist, job_list=shell_seg)
            len_list = []
            len_on_rank = 0
            for idx, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
                    len_i = aop_loc[shell_slice[idx][-1]] - aop_loc[shell_slice[idx][0]]
                    len_list.append(len_i)
                    if idx == rank_i:
                        len_on_rank = len_i
        else:
            shell_slice, len_list = OptPartition(n_rank, shell_seg, naop_seg)
            if len(shell_slice) < n_rank:
                list_app = [None]*(n_rank-len(shell_slice))
                shell_slice.extend(list_app)
                len_list.extend(list_app)
            len_on_rank = len_list[rank_i]
        return shell_slice, len_list, len_on_rank
    def collect_feri(feri_ga):
        t1=time_now()
        win_col = create_win(feri_ga, comm=comm)
        win_col.Fence()
        '''aux_slice = get_slice(range(nrank), job_size=naoaux)
        self.feri_aux_address = [None]*naoaux
        for rank_i, aux_i in enumerate(aux_slice):
            if aux_i is not None:
                idx0 = 0
                for num in aux_i:
                    idx1 = idx0 + 1
                    self.feri_aux_address[num] = [rank_i, [idx0, idx1]]
                    idx0 = idx1
        aux_slice = aux_slice[irank]'''
        aux_slice, self.feri_aux_address = get_auxshell_slice(auxmol)[:2]
        aux_slice = aux_slice[irank]
        if aux_slice is not None:
            #Re-arrange order of data transmission
            idx_break = irank%len(self.feri_aop)
            feri_aop = self.feri_aop[idx_break:] + self.feri_aop[:idx_break]
            max_aop = max([(feri_i[1][-1]-feri_i[1][0]) for feri_i in feri_aop])
            len_aux = len(aux_slice)
            recv_buffer = np.empty(len_aux*max_aop, dtype='f8')
            self.feri_aux = np.empty((len_aux, self.nao_pair), dtype='f8')
            aux0, aux1 = aux_slice[0], aux_slice[-1]+1
            for rank_i, pair_i in feri_aop:
                p0, p1 = pair_i
                win_col.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
                win_col.Get(recv_buffer[:len_aux*(p1-p0)], target_rank=rank_i, target=[aux0*(p1-p0)*8, len_aux*(p1-p0), MPI.DOUBLE])
                win_col.Unlock(rank_i)
                self.feri_aux[:, p0:p1] = recv_buffer[:len_aux*(p1-p0)].reshape(len_aux, -1)
            recv_buffer = None
        else:
            self.feri_aux = None
        win_col.Fence()
        free_win(win_col)
        t1 = time_elapsed(t1)
        log.info('AO int col: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t1[0], t1[1], mem_node(self.pid_list)))

    #def kernel():
    t1=time_now()
    ao_loc = make_loc(self.mol._bas, 'sph')
    aop_loc = ao_loc*(ao_loc+1)//2
    nao = self.nao
    nao_pair = self.nao_pair
    auxmol = df_obj.auxmol
    naoaux = auxmol.nao_nr()
    
    win_low, low_node = get_shared((naoaux, naoaux))
    if irank_shm == 0:
        j2c = auxmol.intor('int2c2e', hermi=1)
        low_node[:] = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=False)
        if irank == 0:
            with h5py.File('j2c_%s.tmp'%meth_type, 'w') as f:
                f.create_dataset('j2c', data=j2c)
                f.create_dataset('low', data=low_node)
    comm_shm.Barrier()

    shell_seg = []
    naop_seg = []
    idx0 = idx1 = 0
    while idx1 < (len(aop_loc)-1):
        idx1 = idx0 + 1
        shell_seg.append([idx0, idx1])
        naop_seg.append(aop_loc[idx1]-aop_loc[idx0])
        idx0 = idx1
    shell_slice, len_list, len_on_rank = get_eri_slice(range(nrank), irank, shell_seg, naop_seg)
    self.feri_aop = []
    for rank_i, shell_i in enumerate(shell_slice):
        if shell_i is not None: 
            s0, s1 = shell_i[0], shell_i[-1]
            feri_idx = [aop_loc[s0], aop_loc[s1]]
            self.feri_aop.append([rank_i, feri_idx])
    shell_slice = shell_slice[irank]
    if (shell_slice is not None):
        #Memory control within a rank
        s0, s1 = shell_slice[0], shell_slice[-1]
        AOP0, AOP1 = aop_loc[s0], aop_loc[s1]
        max_naop = (AOP1-AOP0)
        min_naop = max(naop_seg[s0:s1])
    else:
        max_naop = None
        min_naop = None
    max_naop = get_buff_len(self.mol, size_sub=naoaux, ratio=0.4, max_len=max_naop, min_len=min_naop)
    if (shell_slice is not None):
        int_slice = []
        naop_list = []
        seg_i = []
        naop_i = 0
        shell_seg = shell_seg[s0:s1]
        for idx, si in enumerate(shell_seg):
            s0, s1 = si
            aop0, aop1 = aop_loc[s0], aop_loc[s1]
            if naop_i + (aop1-aop0) > max_naop:
                int_slice.append(seg_i)
                naop_list.append(naop_i)
                seg_i = []
                naop_i = 0
            naop_i += aop1 - aop0
            if seg_i == []:
                seg_i = [s0, s1, aop_loc[s0]-AOP0, aop_loc[s1]-AOP0]
            else:
                seg_i = [seg_i[0], s1, seg_i[2], aop_loc[s1]-AOP0]
            if idx == (len(shell_seg)-1):
                int_slice.append(seg_i)
                naop_list.append(naop_i)
        feri_buffer = np.empty((max(naop_list), naoaux), dtype='f8')
        feri_ga = np.empty((naoaux, len_on_rank), dtype='f8')
        #Generation of 3c2e integrals
        for int_i in int_slice:
            feri_ga = gen_feri(int_i, feri_ga, feri_buffer, low_node)
    else:
        feri_ga = None
    comm.Barrier()
    free_win(win_low); low_node=None
    t1 = time_elapsed(t1)
    log.info('AO int comp: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t1[0], t1[1], mem_node(self.pid_list)))
    collect_feri(feri_ga)
    feri_ga = None

def get_ialp_GA(self, df_obj, meth_type, log, zvec=True):
    def collect_ialp(ialp_aux=None, address_iup=None):
        if self.direct_int:
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            self.ialp_mo_address = [None]*self.no
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i is not None:
                    idx0 = 0
                    for i in mo_i:
                        idx1 = idx0 +1
                        self.ialp_mo_address[i] = [rank_i, [idx0, idx1]]
                        idx0 = idx1
            mo_slice = mo_slice[irank]
            read_atom_close = False

            win_fit, aux_ratio_fit = get_shared((nocc, naoaux))
            if self.chkfile_ialp is None:
                if meth_type == 'hf':
                    self.dir_ijp = 'ijp_hf_tmp'
                    if irank == 0:
                        make_dir(self.dir_ijp)
                if (self.loc_fit):
                    from osvmp2.loc.loc_addons import slice_fit
                    if meth_type == 'hf':
                        win_j2c, j2c_node = get_shared((naoaux, naoaux))
                        if irank_shm == 0:
                            read_file('j2c_%s.tmp'%meth_type, 'j2c', buffer=j2c_node)
                win_low, low_node = get_shared((naoaux, naoaux))
                if irank_shm == 0:
                    read_file('j2c_%s.tmp'%meth_type, 'low', buffer=low_node)
                comm.Barrier()
                if self.use_gpu:
                    with cupy.cuda.Device(self.gpu_id):
                        low_gpu = cupy.asarray(low_node)
                rank_list, seg_list, idx_list = get_GA_slice(address_iup, range(nao))


            if mo_slice is not None:
                mo0, mo1 = mo_slice[0], mo_slice[-1]+1
                nocc_rank = len(mo_slice)
            else:
                nocc_rank = None

            max_mo = get_buff_len(self.mol, size_sub=nao*naoaux, ratio=0.5, max_len=nocc_rank)
            if mo_slice is not None:
                mo_idx = np.append(np.arange(mo0, mo1, step=max_mo), mo1)
                mo_seg = [[mo0, mo1] for mo0, mo1 in zip(mo_idx[:-1], mo_idx[1:])]
                
                if self.loc_fit and meth_type == 'hf':
                    #ialp_i = np.empty((nao, naoaux))
                    buf_ialp_lfit = np.empty(nao*naoaux)
                for mo_i in mo_seg:
                    mo0, mo1 = mo_i
                    nocc_seg = mo1 - mo0
                    if self.chkfile_ialp is None:
                        buf_iup = np.empty(max_mo*nao*naoaux)
                        iup_tmp = buf_iup[:nocc_seg*nao*naoaux].reshape(nao, nocc_seg, naoaux)
                        t1 = time_now()
                        for aidx, rank_i in enumerate(rank_list):
                            ao0, ao1 = seg_list[aidx]
                            file_iup = '%s/%d.tmp'%(self.dir_cal, rank_i)
                            with h5py.File(file_iup, 'r') as f:
                                #f['iup'].read_direct(iup_tmp, source_sel=np.s_[:, mo0:mo1], dest_sel=np.s_[ao0:ao1])
                                f['iup'].read_direct(iup_tmp[ao0:ao1], source_sel=np.s_[:, mo0:mo1])
                        self.t_read += time_now() - t1
                    t1 = time_now()
                    if self.loc_fit and meth_type=='hf' and zvec:
                        auxmol = df_obj.auxmol
                        aux_loc = make_loc(auxmol._bas, 'sph')
                        aux_atm_offset = auxmol.offset_nr_by_atom()
                        for oidx, i in enumerate(range(mo0, mo1)):
                            if self.chkfile_ialp is None:
                                ialp_i = buf_ialp_lfit.reshape(nao, naoaux)
                                ialp_i[:] = iup_tmp[:, oidx]
                                if self.use_gpu:
                                    with cupy.cuda.Device(self.gpu_id):
                                        ialp_gpu = cupy.asarray(ialp_i)
                                        cupyx.scipy.linalg.solve_triangular(low_gpu, ialp_gpu.T, lower=True, overwrite_b=True, 
                                                                            check_finite=False)
                                    ialp_i[:] = cupy.asnumpy(ialp_gpu)
                                else:
                                    scipy.linalg.solve_triangular(low_node, ialp_i.T, lower=True, overwrite_b=True, 
                                                                  check_finite=False)
                                with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialp, i), 'w') as file_ialp:
                                    file_ialp.create_dataset('ialp', shape=(nao, naoaux), dtype='f8')
                                    file_ialp['ialp'].write_direct(ialp_i)
                            else:
                                with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialp, i), "r") as file_ialp:
                                    f['ialp'].read_direct(ialp_i)
                            
                            aux_ratio_fit[i] = np.sum(ialp_i**2, axis=0)
                            atom_close = []
                            for atm, (s0, s1, p0, p1) in enumerate(aux_atm_offset):
                                #if np.mean(aux_ratio_fit[i, p0:p1]) > self.fit_tol:
                                if np.amax(aux_ratio_fit[i, p0:p1]) > self.bfit_tol:
                                    atom_close.append(atm)
                            fit_close = []
                            naux_close = 0
                            for atm0, atm1 in list2seg(atom_close):
                                p0, p1 = aux_atm_offset[atm0][-2], aux_atm_offset[atm1-1][-1]
                                fit_close.append([p0, p1])
                                naux_close += (p1-p0)
                            ialp_lfit = buf_ialp_lfit[:nao*naux_close].reshape(nao, naux_close)
                            ialp_lfit = slice_fit(ialp_lfit, iup_tmp[:, oidx], fit_close, axis=1)
                            ijp_lfit = ddot(self.o.T, ialp_lfit)

                            #Fitting
                            j2c_lfit = np.empty((naux_close, naux_close))
                            j2c_lfit = slice_fit(j2c_lfit, j2c_node, fit_close, axis=2)
                            low_lfit = scipy.linalg.cholesky(j2c_lfit, lower=True, overwrite_a=True)
                            scipy.linalg.solve_triangular(low_lfit, ijp_lfit.T, lower=True, overwrite_b=True, 
                                                          check_finite=False)
                            scipy.linalg.solve_triangular(low_lfit.T, ijp_lfit.T, lower=False, overwrite_b=True, 
                                                          check_finite=False)
                            with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijp, i), 'w') as file_ialp:
                                file_ialp.create_dataset('ijp', shape=(self.no, naux_close), dtype='f8')
                                file_ialp['ijp'].write_direct(ijp_lfit)
                    else:
                        if self.chkfile_ialp is None:
                            iup_tmp = iup_tmp.reshape(-1, naoaux)
                            if self.use_gpu:
                                iup_gpu = cupy.asarray(iup_tmp)
                                cupyx.scipy.linalg.solve_triangular(low_gpu, iup_gpu.T, lower=True, overwrite_b=True, 
                                                              check_finite=False)
                                iup_tmp[:] = cupy.asnumpy(iup_gpu)
                            else:
                                scipy.linalg.solve_triangular(low_node, iup_tmp.T, lower=True, overwrite_b=True, 
                                                            check_finite=False)
                            iup_tmp = iup_tmp.reshape(nao, nocc_seg, naoaux)
                            self.t_ialp += time_now() - t1

                        t1 = time_now()
                        for oidx, i in enumerate(range(mo0, mo1)):
                            if self.chkfile_ialp is None:
                                ialp_i = iup_tmp[:, oidx]
                                with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialp, i), 'w') as file_ialp:
                                    file_ialp.create_dataset('ialp', shape=(nao, naoaux), dtype='f8')
                                    file_ialp['ialp'][:] = ialp_i
                            else:
                                with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialp, i), "r") as file_ialp:
                                    ialp_i = np.asarray(file_ialp['ialp'])
                            if (self.loc_fit) and (read_atom_close is False):
                                aux_ratio_fit[i] = np.sum(ialp_i**2, axis=0)
                            elif meth_type == 'hf':
                                ijp = ddot(self.o.T, ialp_i)
                                scipy.linalg.solve_triangular(low_node.T, ijp.T, lower=False, overwrite_b=True, 
                                                            check_finite=False)
                                with h5py.File('%s/ijp_%d.tmp'%(self.dir_ijp, i), 'w') as file_ialp:
                                    file_ialp.create_dataset('ijp', shape=(self.no, naoaux), dtype='f8')
                                    file_ialp['ijp'].write_direct(ijp)
                            
                        self.t_write += time_now() - t1
            if self.loc_fit:
                from osvmp2.loc.loc_addons import get_fit_domain, get_bfit_domain
                if (meth_type == 'hf') and (self.chkfile_ialp is None):
                    free_win(win_j2c)
                auxmol = df_obj.auxmol
                aux_loc = make_loc(auxmol._bas, 'sph')
                comm.Barrier()
                Acc_and_get_GA(aux_ratio_fit)
                self.fit_list, self.fit_seg, self.nfit = get_fit_domain(mol, auxmol, aux_ratio_fit, self.fit_tol)
                self.atom_close, self.bfit_seg, self.nbfit, self.cal_seg = get_bfit_domain(mol, auxmol, aux_ratio_fit, self.fit_tol, use_group=False)
                if meth_type == 'hf' and zvec:
                    self.atom_close_z, self.bfit_seg_z, self.nbfit_z, self.cal_seg_z = get_bfit_domain(mol, auxmol, aux_ratio_fit, self.bfit_tol, use_group=False)
                #self.fit_seg, self.fit_list, self.nfit, self.atom_close, self.fit_atom, self.nfit_atom, self.cal_seg = get_fit(aux_ratio_fit, self.fit_tol)
            else:
                free_win(win_low); low_node=None
        else:
            if ialp_aux is not None:
                #ialp_aux = contigous_trans(ialp_aux, (1,0,2))
                ialp_aux = contigous_trans(ialp_aux, (2,1,0))
            win_col = create_win(ialp_aux, comm=comm)
            win_col.Fence()
            #mo_slice = get_slice(range(nrank), job_size=self.no)
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            self.ialp_mo_address = [None]*self.no
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i is not None:
                    for idx, num in enumerate(mo_i):
                        self.ialp_mo_address[num] = [rank_i, [idx, idx+1]]
            mo_slice = mo_slice[irank]
            if mo_slice is not None:
                rank_list, seg_list, idx_list = get_GA_slice(self.ialp_aux_address, range(self.naoaux))
                job_idxlist = list(range(len(rank_list)))
                idx_break = irank%len(job_idxlist)
                job_idxlist = job_idxlist[idx_break:] + job_idxlist[:idx_break]
                len_mo = len(mo_slice)
                max_aux = max([(seg_i[-1]-seg_i[0]) for seg_i in seg_list])
                recv_buffer = np.empty(len_mo*max_aux*self.nao, dtype='f8')
                #self.ialp_mo = np.empty((len_mo, self.naoaux, self.nao), dtype='f8')
                self.ialp_mo = np.empty((len_mo, self.nao, self.naoaux), dtype='f8')
                mo0, mo1 = mo_slice[0], mo_slice[-1]+1
                for j_idx in job_idxlist:
                    rank_i = rank_list[j_idx]
                    aux0, aux1 = seg_list[j_idx]
                    f_idx0, f_idx1 = idx_list[j_idx]
                    win_col.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
                    win_col.Get(recv_buffer[:len_mo*(aux1-aux0)*self.nao], target_rank=rank_i, target=[mo0*(aux1-aux0)*self.nao*8, len_mo*(aux1-aux0)*self.nao, MPI.DOUBLE])
                    win_col.Unlock(rank_i)
                    #self.ialp_mo[:, aux0:aux1] = recv_buffer[:len_mo*(aux1-aux0)*self.nao].reshape(len_mo, -1, self.nao)
                    self.ialp_mo[:, :, aux0:aux1] = recv_buffer[:len_mo*(aux1-aux0)*self.nao].reshape(len_mo, self.nao, -1)
                recv_buffer = None
            else:
                self.ialp_mo = None
                self.ialp_aux = None
            win_col.Fence(); free_win(win_col)
            self.win_ialp_mo = create_win(self.ialp_mo, comm=comm)
            self.win_ialp_mo.Fence()

            self.dir_ialp = 'ialp_mo'
            os.makedirs(self.dir_ialp, exist_ok=True)
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)[irank]
            if mo_slice is not None:
                for idx_i, i in enumerate(mo_slice):
                    with h5py.File('%s/ialp_%d.tmp'%(self.dir_ialp, i), 'w') as file_ialp:
                        file_ialp.create_dataset('ialp', shape=(self.nao, self.naoaux), dtype='f8')
                        file_ialp['ialp'].write_direct(self.ialp_mo[idx_i])
            if (self.grad_cal) and (self.direct_int is False) and (meth_type=='hf') and (ialp_aux is not None):
                #self.ialp_aux = ialp_aux
                #self.ialp_aux = contigous_trans(ialp_aux, (1,0,2))
                self.ialp_aux = contigous_trans(ialp_aux, (2,1,0))
    if (self.direct_int):
        self.t_feri = np.zeros(2)
        self.t_ialp = np.zeros(2)
        self.t_read = np.zeros(2)
        self.t_write = np.zeros(2)
        #t1 = time_now()
        tt=time_now() 
        nao = self.mol.nao_nr()
        nocc = self.mol.nelectron//2
        naoaux = self.naoaux
        mol = self.mol
        auxmol = df_obj.auxmol
        ao_loc = make_loc(self.mol._bas, 'sph')
        self.dir_ialp = "ialp_mo_%s_tmp"%meth_type
        #comm.Barrier()
        def get_iup():
            self.dir_cal = "%s/ialp_cal_tmp"%self.dir_ialp
            if irank == 0:
                os.makedirs(self.dir_cal, exist_ok=True)
            comm.Barrier()
            ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(mol, self.shell_slice, aslice=True)
            address_iup = [None]*(nao+1)
            for rank_i, slice_i in enumerate(ao_slice):
                if slice_i is not None:
                    ao0, ao1 = slice_i
                    for idx, aox in enumerate(range(ao0, ao1)):
                        address_iup[aox] = [rank_i, [idx, idx+1]]

        
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            self.ialp_mo_address = [None]*self.no
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i is not None:
                    for idx, i in enumerate(mo_i):
                        self.ialp_mo_address[i] = [rank_i, [idx, idx+1]]
            max_memory = get_mem_spare(mol, 0.9)
            if shell_slice_rank is not None:
                size_ialp, size_feri, shell_slice_rank = mem_control(mol, nocc, naoaux, shell_slice_rank, 
                                                                     "half_trans", max_memory)
                buf_ialp = np.empty(size_ialp)
                buf_feri = np.empty(size_feri)
                nao_rank = ao_slice[irank][1] - ao_slice[irank][0]
                with h5py.File("%s/%d.tmp"%(self.dir_cal, irank), 'w') as file_iup:
                    file_iup.create_dataset('iup', (nao_rank, nocc, naoaux), dtype='f8')
                SHELL_SEG = slice2seg(mol, shell_slice_rank, max_nao=buf_ialp.size//(nocc*naoaux))
                ao_idx0 = 0
                if self.use_gpu:
                    mo_gpu = cupy.asarray(self.o)
                for seg_i in SHELL_SEG:
                    A0, A1 = seg_i[0][0], seg_i[-1][1]
                    AL0, AL1 = ao_loc[A0], ao_loc[A1]
                    nao_seg = AL1 - AL0
                    iup_tmp = buf_ialp[:nao_seg*nocc*naoaux].reshape(nao_seg, nocc, naoaux)
                    iup_tmp[:] = 0
                    buf_idx0 = 0
                    for a0, a1, b_list in seg_i:
                        al0, al1 = ao_loc[a0], ao_loc[a1]
                        nao0 = al1 - al0
                        buf_idx1 = buf_idx0 + nao0
                        for b0, b1 in b_list:
                            be0, be1 = ao_loc[b0], ao_loc[b1]
                            nao1 = be1 - be0
                            s_slice = (a0, a1, b0, b1, mol.nbas, mol.nbas+auxmol.nbas)
                            t1 = time_now()
                            feri_tmp = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri).transpose(1,0,2) #(nao1, nao0, naoaux)
                            self.t_feri += time_now() - t1

                            t1 = time_now()
                            if self.use_gpu:
                                #nao1*nao2*naoaux + nocc*nao2*naoaux
                                max_nao_gpu = int(0.8*(self.gpu_memory*1e6) // (8*(nao1 + nocc)*naoaux))
                                max_nao_gpu = max(1, min(nao0, max_nao_gpu))
                                mo_gpu_step = mo_gpu[be0:be1].T
                                bidx0 = buf_idx0
                                gidx0 = 0
                                for ga0 in np.arange(al0, al1, step=max_nao_gpu):
                                    ga1 = min(al1, ga0+max_nao_gpu)
                                    nao2 = ga1 - ga0
                                    bidx1 = bidx0 + nao2
                                    gidx1 = gidx0 + nao2
                                    feri_gpu = cupy.asarray(feri_tmp[:, gidx0:gidx1])
                                    iup_gpu = cupy.dot(mo_gpu_step, feri_gpu.reshape(nao1, -1))
                                    iup_tmp[bidx0:bidx1] += cupy.asnumpy(iup_gpu).reshape(nocc, -1, naoaux).transpose(1,0,2)
                                    bidx0 = bidx1
                                    gidx0 = gidx1
                            else:
                                iup_tmp[buf_idx0:buf_idx1] += ddot(self.o[be0:be1].T, feri_tmp.reshape(nao1, -1)).reshape(nocc, nao0, naoaux).transpose(1,0,2)
                            self.t_ialp += time_now() - t1
                        buf_idx0 = buf_idx1
                    ao_idx1 = ao_idx0 + nao_seg
                    t1 = time_now()
                    with h5py.File("%s/%d.tmp"%(self.dir_cal, irank), 'r+') as file_iup:
                        try:
                            file_iup['iup'].write_direct(iup_tmp, dest_sel=np.s_[ao_idx0:ao_idx1])
                        except TypeError:
                            print(irank, ao_slice[irank], iup_tmp.shape, file_iup['iup'].shape, ao_idx0, ao_idx1, shell_slice_rank);sys.exit()
                    self.t_write += time_now() - t1
                    ao_idx0 = ao_idx1
                buf_ialp, iup_tmp = None, None
                
            comm.Barrier()
            #self.loc_fit = False
            return address_iup


        #Get j2c
        if irank == 0:
            t1=time_now() 
            if os.path.isfile('j2c_%s.tmp'%meth_type) is False:
                with h5py.File('j2c_%s.tmp'%meth_type, 'w') as f:
                    j2c = auxmol.intor('int2c2e', hermi=1)
                    f.create_dataset('j2c', data=j2c)
                    low = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
                    f.create_dataset('low', data=low)
                    j2c, low = None, None
                print_time(['Fitting int', time_elapsed(t1)], log)
        comm.Barrier()

        if self.chkfile_ialp is None:
            t1=time_now() 
            address_iup = get_iup()
            print_time(['computing ialp', time_elapsed(t1)], log)
        else:
            address_iup = None
            self.dir_ialp = self.chkfile_ialp
            log.info(f"Use ialp check file: {self.chkfile_ialp}")


        t1=time_now() 
        collect_ialp(address_iup=address_iup)
        comm.Barrier()
        if irank == 0:
            print_time(['ialq and collection', time_elapsed(t1)], log)
            time_list = [['feri', self.t_feri], ['ialp', self.t_ialp],
                            ['reading', self.t_read], ['writing', self.t_write]]
            print_time(time_list, log)
            if self.chkfile_ialp is None:
                shutil.rmtree(self.dir_cal)
        
    else:
        t1=time_now() 
        '''aux_slice = get_slice(range(nrank), job_size=self.naoaux)
        self.ialp_aux_address = [None]*self.naoaux
        for rank_i, aux_i in enumerate(aux_slice):
            if aux_i is not None:
                idx0 = 0
                for num in aux_i:
                    idx1 = idx0 +1
                    self.ialp_aux_address[num] = [rank_i, [idx0, idx1]]
                    idx0 = idx1
        aux_slice = aux_slice[irank]'''
        aux_slice, self.ialp_aux_address = get_auxshell_slice(df_obj.auxmol)[:2]
        aux_slice = aux_slice[irank]
        if aux_slice is not None:
            #ialp_aux = np.empty((len(aux_slice), self.no, self.nao), dtype='f8')
            ialp_aux = np.empty((len(aux_slice), self.nao, self.no), dtype='f8')
            feri_buffer_unpack = np.empty((self.nao, self.nao))
            if (self.outcore):
                with h5py.File(self.feri_aux, 'r') as feri_aux:
                    for idx, num in enumerate(aux_slice):
                        lib.numpy_helper.unpack_tril(np.asarray(feri_aux['j3c'][idx]), out=feri_buffer_unpack)
                        #ialp_aux[idx] = ddot(feri_buffer_unpack, self.o).T
                        ialp_aux[idx] = ddot(feri_buffer_unpack, self.o)
            else:
                for idx, num in enumerate(aux_slice):
                    lib.numpy_helper.unpack_tril(self.feri_aux[idx], out=feri_buffer_unpack)
                    #ialp_aux[idx] = ddot(feri_buffer_unpack, self.o).T
                    ialp_aux[idx] = ddot(feri_buffer_unpack, self.o)
            if irank == 0:
                print_time(['computing ialp', time_elapsed(t1)], log)
            if self.grad_cal is False:
                self.feri_aux = None
        else:
            ialp_aux = None
        comm.Barrier()
        t1=time_now() 
        collect_ialp(ialp_aux)
        if irank == 0:
            print_time(['collecting ialp', time_elapsed(t1)], log)



def OSV_generation_GA(self, log):
    def get_osv(idx, i, ialp_i, use_idsvd=True):
        if self.idsvd_tol < 0:
            use_idsvd = False
        import scipy.linalg.interpolative as sli
        if self.chkfile_ti is None:
            t1 = time_now()
            #iap = ddot(self.v.T, ialp_i)
            moint_ovv = multi_dot([self.v.T, ialp_i, ialp_i.T, self.v])
            #moint_ovv = ddot(iap, iap.T)
            bottom = 1.0/(-2*self.eo[i] + (self.ev + self.ev.reshape(-1, 1)))
            ti = moint_ovv * bottom
            self.t_ti += time_now() - t1
        else:
            with h5py.File('%s/ti_%d.tmp'%(self.dir_ti, i), 'r') as file_ti:
                ti = np.asarray(file_ti["ti"])

        #Singular value decomposition of T_ii
        t1 = time_now()
        if self.chkfile_qcp is None:
            if use_idsvd:
                _RETCODE_ERROR = RuntimeError("nonzero return code")
                if self.nosv_id is not None:
                    qcp, s, v = sli.svd(ti, self.nosv_id)
                else:
                    try:
                        qcp, s, v = sli.svd(ti, self.idsvd_tol)
                    except RuntimeError:
                        qcp, s, v = sli.svd(ti, min(100, self.nv))
                    if self.ml_test and len(s) < 8:
                        qcp, s, v = sli.svd(ti, 8)
            else:
                qcp, s, v = svd(ti)
                nosv_cp = sum(s >= self.cposv_tol)
                if nosv_cp < 8:
                    nosv_cp = 8
                qcp = qcp[:, :nosv_cp]
                s = s[:nosv_cp]
            # print("qcp ti", i, np.linalg.norm(qcp), np.linalg.norm(ti))
        else:
            with h5py.File('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'r') as file_ti:
                qcp = np.asarray(file_ti["qcp"])
                s = np.asarray(file_ti["s"])
        self.nosv_cp[i] = len(s)
        if self.nosv_cp[i] == 0:
            self.nosv_cp[i] = 1
        self.nosv[i] = sum(s >= self.osv_tol)
        if self.nosv[i] == 0:
            self.nosv[i] = 1
        if self.ml_test and self.nosv[i] < 8:
            self.nosv[i] = 8
        self.s_ga[idx, :self.nosv_cp[i]] = s
        self.t_svd += time_now() - t1

        t1 = time_now()
        if self.chkfile_qcp is None:
            with h5py.File('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'w') as file_qcp:
                file_qcp.create_dataset('qcp', shape=(self.nv, self.nosv_cp[i]), dtype='f8')
                file_qcp['qcp'][...] = qcp
                file_qcp['s'] = s
        #if self.grad_cal and self.chkfile_ti is None:
        if self.chkfile_ti is None:
            with h5py.File('%s/ti_%d.tmp'%(self.dir_ti, i), 'w') as file_ti:
                file_ti.create_dataset('ti', shape=(self.nv, self.nv), dtype='f8')
                file_ti['ti'].write_direct(ti)
        self.t_write += time_now() - t1

    mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))
    self.mo_address = [None]*self.no
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            idx0 = 0
            for i in mo_i:
                idx1 = idx0 + 1
                self.mo_address[i] = [rank_i, [idx0, idx1]]
                idx0 = idx1
    mo_slice = mo_slice[irank]
    self.win_nosv, self.nosv = get_shared(self.no, dtype='i')
    self.win_nosv_cp, self.nosv_cp = get_shared(self.no, dtype='i')
    if (self.chkfile_ti is not None):
        self.dir_ti = self.chkfile_ti
    else:
        self.dir_ti = 'ti_tmp'
        if irank == 0:
            make_dir(self.dir_ti)
    if (self.chkfile_qcp is not None):
        self.dir_qcp = self.chkfile_qcp
    else:    
        self.dir_qcp = 'qcp_tmp'
        if irank == 0:
            make_dir(self.dir_qcp)
    comm.Barrier()
    self.t_ti = np.zeros(2)
    self.t_svd = np.zeros(2)
    self.t_read = np.zeros(2)
    self.t_write = np.zeros(2)
    self.t_slice = np.zeros(2)
    if mo_slice is not None:
        len_mo = len(mo_slice)
        self.s_ga = np.zeros((len_mo, self.nv), dtype='f8')
        buf_ialp = np.empty((self.nao, self.naoaux))
        if self.loc_fit:
            max_aux = max([self.nfit[i] for i in mo_slice])
            buf_ialp_loc = np.empty((self.nao*max_aux))            
        if self.direct_int:
            for idx, i in enumerate(mo_slice):
                file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                if self.loc_fit:
                    if self.chkfile_ti is None:
                        naux_i = self.nfit[i]
                        ialp_tmp = buf_ialp_loc[:self.nao*naux_i].reshape(self.nao, naux_i)
                        t1 = time_now()
                        read_file(file_ialp, 'ialp', buffer=buf_ialp)
                        self.t_read += time_now() - t1

                        t1 = time_now()
                        idx_p0 = 0
                        for p0, p1 in self.fit_seg[i]:
                            idx_p1 = idx_p0 + (p1-p0)
                            ialp_tmp[:, idx_p0:idx_p1] = buf_ialp[:, p0:p1]
                            idx_p0 = idx_p1
                        #ialp_tmp = buf_ialp[:, self.fit_list[i]]
                        self.t_slice += time_now() - t1
                    else:
                        ialp_tmp = None
                    get_osv(idx, i, ialp_tmp)
                else:
                    if self.chkfile_ti is None:
                        t1 = time_now()
                        read_file(file_ialp, 'ialp', buffer=buf_ialp)
                        self.t_read += time_now() - t1
                    get_osv(idx, i, buf_ialp)
        else:
            for idx, i in enumerate(mo_slice):
                # print("ialp_mo", i, np.linalg.norm(self.ialp_mo[idx]))
                get_osv(idx, i, self.ialp_mo[idx])
            if self.grad_cal:
                self.qcp_ga = np.empty(sum([self.nv*self.nosv_cp[i] for i in mo_slice]))
        if self.idsvd_tol >= 0:
            log.info('    Threshold of id-SVD: %.2E'%self.idsvd_tol)
        t1 = time_now()
        self.qmat_ga = np.empty(sum([self.nv*self.nosv[i] for i in mo_slice]))
        idx_qcp0 = 0
        idx_qmat0 = 0
        for idx, i in enumerate(mo_slice):
            idx_qcp1 = idx_qcp0 + self.nv*self.nosv_cp[i]
            idx_qmat1 = idx_qmat0 + self.nv*self.nosv[i]
            #if self.direct_int:
            if (self.grad_cal) and (self.direct_int is False):
                buf_qcp = self.qcp_ga[idx_qcp0: idx_qcp1].reshape(self.nv, self.nosv_cp[i])
                with h5py.File('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'r') as file_qcp:
                    file_qcp['qcp'].read_direct(buf_qcp)
                self.qmat_ga[idx_qmat0: idx_qmat1] = buf_qcp[:, :self.nosv[i]].ravel()
            else:
                buf_qmat = self.qmat_ga[idx_qmat0: idx_qmat1].reshape(self.nv, self.nosv[i])
                with h5py.File('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'r') as file_qcp:
                    file_qcp['qcp'].read_direct(buf_qmat, np.s_[:, :self.nosv[i]])
            idx_qcp0 = idx_qcp1
            idx_qmat0 = idx_qmat1
        self.t_read += time_now() - t1
    else:
        self.s_ga = None
        self.qmat_ga = None
        self.qcp_ga = None
    comm.Barrier()
    Acc_and_get_GA(self.nosv_cp)
    Acc_and_get_GA(self.nosv)
    mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))
    self.dim_qcp = [None]*self.no
    self.dim_qmat = [None]*self.no
    self.qcp_address = [None]*self.no
    self.qmat_address = [None]*self.no
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            idx_qcp0 = 0
            idx_qmat0 = 0
            for idx, i in enumerate(mo_i):
                self.dim_qcp[i] = (self.nv, self.nosv_cp[i])
                self.dim_qmat[i] = (self.nv, self.nosv[i])
                idx_qcp1 = idx_qcp0 + self.nv*self.nosv_cp[i]
                idx_qmat1 = idx_qmat0 + self.nv*self.nosv[i]
                self.qcp_address[i] = [rank_i, [idx_qcp0, idx_qcp1]]
                self.qmat_address[i] = [rank_i, [idx_qmat0, idx_qmat1]]
                idx_qcp0 = idx_qcp1
                idx_qmat0 = idx_qmat1
    self.win_qmat = create_win(self.qmat_ga, comm=comm)
    self.win_qmat.Fence()
    comm.Barrier()
    
    if (self.grad_cal) and (self.direct_int is False):
        self.win_qcp = create_win(self.qcp_ga, comm=comm)
        self.win_qcp.Fence()
    else:
        self.qcp_ga = None
    time_list = [['T_ii', self.t_ti], ['SVD', self.t_svd],
                 ['reading', self.t_read], ['writing', self.t_write]]
    if self.loc_fit:
        time_list.append(['slicing', self.t_slice])
    print_time(time_list, log)
    #sys.exit()

def update_qmat_ml(self):
    mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))
    nosv_new = [self.nosv_ml]*self.no
    qmat_address_new = [None]*self.no
    dim_qmat_new = [None]*self.no
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            idx_qmat0 = 0
            for idx, i in enumerate(mo_i):
                idx_qmat1 = idx_qmat0 + self.nv*self.nosv_ml
                dim_qmat_new[i] = (self.nv, self.nosv_ml)
                qmat_address_new[i] = [rank_i, [idx_qmat0, idx_qmat1]]
                idx_qmat0 = idx_qmat1
    free_win(self.win_qmat)
    comm.Barrier()
    mo_slice = mo_slice[irank]
    if mo_slice is not None:
        qmat_ga_new = np.empty(self.nv*self.nosv_ml*len(mo_slice))
        new_idx0 = 0
        for i in mo_slice:
            rank_i, (idx_qmat0, idx_qmat1) = self.qmat_address[i]
            new_idx1 = new_idx0 + self.nv*self.nosv_ml
            qmat_old = self.qmat_ga[idx_qmat0:idx_qmat1].reshape(self.nv, self.nosv[i])
            idx1_save = new_idx0 + self.nv*min(self.nosv_ml, self.nosv[i])
            qmat_ga_new[new_idx0:idx1_save] = qmat_old[:, :self.nosv_ml].ravel()
            new_idx0 = new_idx1
    else:
        qmat_ga_new = None
    self.nosv = nosv_new
    self.dim_qmat = dim_qmat_new
    self.qmat_address = qmat_address_new
    self.qmat_ga = qmat_ga_new
    self.win_qmat = create_win(self.qmat_ga, comm=comm)
    self.win_qmat.Fence()
    comm.Barrier()

def update_sf_ml(self):
    dim_sf_new = [(self.nosv_ml, self.nosv_ml)]*self.no**2
    sf_address_new = [None]*self.no**2
    pair_slice = get_pairslice(self, pairlist=self.pairlist)
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            sf_idx0 = 0
            for ipair in pair_i:
                sf_idx1 = sf_idx0 + self.nosv_ml**2
                sf_address_new[ipair] = [rank_i, [sf_idx0, sf_idx1]]
                sf_idx0 = sf_idx1
    pair_slice = pair_slice[irank]
    if pair_slice is not None:
        smat_ga_new = np.empty(len(pair_slice)*self.nosv_ml**2, dtype='f8')
        fmat_ga_new = np.copy(self.smat_ga)
        max_fdim = max([np.prod(self.dim_sf[ipair]) for ipair in pair_slice])
        sf_buffer = np.empty(max_fdim, dtype='f8')
        sf_tmp = [None]*self.no**2
        new_idx0 = 0
        for ipair in pair_slice:
            new_idx1 = new_idx0 + self.nosv_ml**2
            sf_buffer, sf_tmp = read_GA(self.sf_address, [ipair], sf_buffer, self.win_smat, dtype='f8', list_col=sf_tmp, dim_list=self.dim_sf)
            smat_ga_new[new_idx0:new_idx1] = sf_tmp[ipair][:self.nosv_ml, :self.nosv_ml].ravel()
            sf_tmp[ipair] = None
            sf_buffer, sf_tmp = read_GA(self.sf_address, [ipair], sf_buffer, self.win_fmat, dtype='f8', list_col=sf_tmp, dim_list=self.dim_sf)
            fmat_ga_new[new_idx0:new_idx1] = sf_tmp[ipair][:self.nosv_ml, :self.nosv_ml].ravel()
            sf_tmp[ipair] = None
            new_idx0 = new_idx1
    else:
        smat_ga_new = None
        fmat_ga_new = None
    comm.Barrier()
    for win in [self.win_smat, self.win_fmat]:
        free_win(win)
    self.dim_sf = dim_sf_new
    self.sf_address = sf_address_new
    self.smat_ga = smat_ga_new
    self.fmat_ga = fmat_ga_new
    self.win_smat = create_win(self.smat_ga, comm=comm)
    self.win_fmat = create_win(self.fmat_ga, comm=comm)
    for win in [self.win_smat, self.win_fmat]:
        win.Fence()
    comm.Barrier()

def get_sf_cp_GA(self):
    #pair_slice = get_slice(job_list=self.pairlist_full, rank_list=range(nrank))
    if self.direct_int:
        self.dir_sf_cp = 'sf_cp_tmp'
        if irank == 0:
            make_dir(self.dir_sf_cp)
        comm.Barrier()
    pair_slice = get_pairslice(self, if_remote=False, if_full=True)
    self.dim_sf_cp = [None]*self.no**2
    self.sf_cp_address = [None]*self.no**2
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            sf_idx0 = 0
            for ipair in pair_i:
                i = ipair // self.no
                j = ipair % self.no
                self.dim_sf_cp[ipair] = (self.nosv_cp[i], self.nosv[j])
                sf_idx1 = sf_idx0 + self.nosv_cp[i]*self.nosv[j]
                self.sf_cp_address[ipair] = [rank_i, [sf_idx0, sf_idx1]]
                sf_idx0 = sf_idx1
    pair_slice = pair_slice[irank]
    if pair_slice is not None:
        dim_qmat = []
        dim_qcp = []
        dim_sf_cp = []
        for ipair in pair_slice:
            i = ipair // self.no
            j = ipair % self.no
            dim_qmat.append(self.nv*self.nosv[j])
            dim_qcp.append(self.nv*self.nosv_cp[i])
            dim_sf_cp.append(self.nosv_cp[i]*self.nosv[j])
            
        buf_qmat = np.empty(max(dim_qmat), dtype='f8')
        buf_qcp = np.empty(max(dim_qcp), dtype='f8')
        #qmat_tmp = [None]*self.no
        if self.direct_int:
            file_sf_cp = h5py.File("%s/sf_cp_%d.tmp"%(self.dir_sf_cp, irank), 'w')
            scp_save = file_sf_cp.create_dataset('scp', (sum(dim_sf_cp), ), dtype='f8')
            fcp_save = file_sf_cp.create_dataset('fcp', (sum(dim_sf_cp), ), dtype='f8')
            
        else:
            self.scp_ga = np.empty(sum(dim_sf_cp), dtype='f8')
            self.fcp_ga = np.copy(self.scp_ga)
        i_pre, j_pre = 0, 0
        for idx, ipair in enumerate(pair_slice):
            i = ipair // self.no
            j = ipair % self.no
            #buf_qmat, qmat_tmp = read_GA(self.qmat_address, [j], buf_qmat, self.win_qmat, dtype='f8', list_col=qmat_tmp, dim_list=self.dim_qmat)
            qmat_tmp = buf_qmat[:self.nv*self.nosv[j]]
            qmat_tmp = read_GA(self.qmat_address, [j], qmat_tmp, self.win_qmat, dtype='f8').reshape(self.nv, self.nosv[j])
            if (idx == 0) or (i != i_pre):
                #if self.direct_int:
                #read_file('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'qcp', buffer=qcp_tmp)
                qcp_tmp = buf_qcp[:self.nv*self.nosv_cp[i]].reshape(self.nv, self.nosv_cp[i])
                read_file('%s/qcp_%d.tmp'%(self.dir_qcp, i), 'qcp', buffer=qcp_tmp)
                '''else:
                    qcp_tmp = buf_qcp[:self.nv*self.nosv_cp[i]]
                    qcp_tmp = read_GA(self.qcp_address, [i], qcp_tmp, self.win_qcp, dtype='f8').reshape(self.nv, self.nosv_cp[i])'''
            sf_idx0, sf_idx1 = self.sf_cp_address[ipair][1]
            if self.direct_int:
                scp_save[sf_idx0: sf_idx1] = ddot(qcp_tmp.T, qmat_tmp).ravel()
                fcp_save[sf_idx0: sf_idx1] = ddot(np.multiply(qcp_tmp.T, self.ev), qmat_tmp).ravel()
                #fcp_save[sf_idx0: sf_idx1] = multi_dot([qcp_tmp.T, self.ev_di, qmat_tmp]).ravel()
            else:
                self.scp_ga[sf_idx0: sf_idx1] = ddot(qcp_tmp.T, qmat_tmp).ravel()
                #self.fcp_ga[sf_idx0: sf_idx1] = multi_dot([qcp_tmp.T, self.ev_di, qmat_tmp]).ravel()
                self.fcp_ga[sf_idx0: sf_idx1] = ddot(np.multiply(qcp_tmp.T, self.ev), qmat_tmp).ravel()
            #qmat_tmp[j] = None
            i_pre, j_pre = i, j
        
    else:
        self.scp_ga = None
        self.fcp_ga = None
    if self.direct_int:
        if pair_slice is not None:
            file_sf_cp.close()
        
    else:
        self.win_scp = create_win(self.scp_ga, comm=comm)
        self.win_fcp = create_win(self.fcp_ga, comm=comm)
        self.win_scp.Fence()
        self.win_fcp.Fence()   

def get_sf_GA(self):
    #pair_slice = get_slice(job_list=self.pairlist, rank_list=range(nrank))
    pair_slice = get_pairslice(self, pairlist=self.pairlist)
    self.dim_sf = [None]*self.no**2
    self.sf_address = [None]*self.no**2
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            sf_idx0 = 0
            for ipair in pair_i:
                i = ipair // self.no
                j = ipair % self.no
                self.dim_sf[ipair] = (self.nosv[i], self.nosv[j])
                sf_idx1 = sf_idx0 + self.nosv[i]*self.nosv[j]
                self.sf_address[ipair] = [rank_i, [sf_idx0, sf_idx1]]
                sf_idx0 = sf_idx1
    pair_slice = pair_slice[irank]
    if pair_slice is not None:
        qmat_dim = []
        sf_dim = []
        for ipair in pair_slice:
            i = ipair // self.no
            j = ipair % self.no
            qmat_dim.append(self.nv*(self.nosv[i]+self.nosv[j]))
            sf_dim.append(self.nosv[i]*self.nosv[j])

        qmat_buffer = np.empty(max(qmat_dim), dtype='f8')
        Q_matrix = [None]*self.no
        self.smat_ga = np.empty(sum(sf_dim), dtype='f8')
        self.fmat_ga = np.copy(self.smat_ga)
        if self.ml_test:
            e_ab = self.ev + self.ev.reshape(-1, 1)
        sf_idx0 = 0
        for idx, ipair in enumerate(pair_slice):
            i = ipair // self.no
            j = ipair % self.no
            #Q_matrix = read_GA_node([i, j], self.qmat_address_node, self.qmat_node, self.dim_qmat, Q_matrix, qmat_buffer)
            qmat_buffer, Q_matrix = read_GA(self.qmat_address, [i, j], qmat_buffer, self.win_qmat, dtype='f8', list_col=Q_matrix, dim_list=self.dim_qmat)
            sf_idx1 = sf_idx0 + np.product(self.dim_sf[ipair])
            self.smat_ga[sf_idx0:sf_idx1] = ddot(Q_matrix[i].T, Q_matrix[j]).ravel()
            s_tmp = self.smat_ga[sf_idx0:sf_idx1]
            #self.fmat_ga[sf_idx0:sf_idx1] = multi_dot([Q_matrix[i].T, self.ev_di, Q_matrix[j]]).ravel()
            self.fmat_ga[sf_idx0:sf_idx1] = ddot(np.multiply(Q_matrix[i].T, self.ev), Q_matrix[j]).ravel()
            sf_idx0 = sf_idx1
            if ipair != pair_slice[-1]:
                if i != j:
                    Q_matrix[j] = None
                pair_next = pair_slice[idx+1]
                i_next = pair_next//self.no
                if i_next != i:
                    Q_matrix[i] = None
        qmat_buffer = None
        Q_matrix = None
    else:
        self.smat_ga = None
        self.fmat_ga = None
    comm.Barrier()
    self.win_smat = create_win(self.smat_ga, comm=comm)
    self.win_fmat = create_win(self.fmat_ga, comm=comm)
    self.win_smat.Fence()
    self.win_fmat.Fence()

def get_sratio_GA(self):
    self.win_s_r, self.s_ratio = get_shared(self.no**2, dtype='f8')
    #pair_slice = get_slice(job_list=self.pairlist, rank_list=range(nrank))[irank]
    pair_slice = get_pairslice(self, pairlist=self.pairlist)[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            i = ipair // self.no
            j = ipair % self.no
            loc0, loc1 = self.sf_address[ipair][-1]
            s_tmp = self.smat_ga[loc0:loc1]
            pair_ji = j*self.no+i
            self.s_ratio[pair_ji] = self.s_ratio[ipair] = sum((s_tmp**2)/((self.nosv[i]+self.nosv[j])*0.5))
    comm.Barrier()
    if irank_shm == 0:
        win_col = create_win(self.s_ratio, comm=comm)
    else:
        win_col = create_win(None, comm=comm)
    win_col.Fence()
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0)
        win_col.Accumulate(self.s_ratio, target_rank=0, op=MPI.SUM)
        win_col.Unlock(0)
    win_col.Fence()
    if irank_shm == 0 and irank != 0:
        win_col.Lock(0)
        win_col.Get(self.s_ratio, target_rank=0)
        win_col.Unlock(0)
    win_col.Fence()
    free_win(win_col)

def get_precon_by_mo_GA(self):
    mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))
    self.dim_xii = [None]*self.no
    self.dim_emui = [None]*self.no
    self.xii_address = [None]*self.no
    self.emui_address = [None]*self.no
    dim_xii = []
    dim_emui = []
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            idx_xii0, idx_emui0 = 0, 0
            for idx, i in enumerate(mo_i):
                self.dim_xii[i] = (self.nosv[i], self.nosv[i])
                idx_xii1 = idx_xii0 + self.nosv[i]**2
                self.xii_address[i] = [rank_i, [idx_xii0, idx_xii1]]
                self.dim_emui[i] = self.nosv[i]
                idx_emui1 = idx_emui0 + self.nosv[i]
                self.emui_address[i] = [rank_i, [idx_emui0, idx_emui1]]
                if rank_i == irank:
                    dim_xii.append(self.nosv[i]**2)
                    dim_emui.append(self.nosv[i])
                idx_xii0, idx_emui0 = idx_xii1, idx_emui1
    mo_slice = mo_slice[irank]
    if mo_slice is not None:
        self.xii_ga = np.empty(sum(dim_xii), dtype='f8')
        self.emui_ga = np.empty(sum(dim_emui), dtype='f8')
        max_fdim = max([self.nosv[i]**2 for i in mo_slice])
        f_buffer = np.empty(max_fdim, dtype='f8')
        idx_xii0, idx_emui0 = 0, 0
        for i in mo_slice:
            ii = i*self.no + i
            rank_i, f_idx = self.sf_address[ii]
            f_idx0, f_idx1 = f_idx
            f_dim = self.nosv[i]**2
            self.win_fmat.Lock(rank_i, lock_type=MPI.LOCK_SHARED)
            self.win_fmat.Get(f_buffer[:f_dim], target_rank=rank_i, target=[f_idx0*8, (f_idx1-f_idx0), MPI.DOUBLE])
            self.win_fmat.Unlock(rank_i)
            f_ii = f_buffer[:f_dim].reshape(self.nosv[i], self.nosv[i])
            idx_xii1 = idx_xii0 + self.nosv[i]**2
            idx_emui1 = idx_emui0 + self.nosv[i]
            emui, xii = eigh(f_ii)
            self.emui_ga[idx_emui0: idx_emui1] = emui.ravel()
            self.xii_ga[idx_xii0: idx_xii1] = xii.ravel()
            idx_xii0, idx_emui0 = idx_xii1, idx_emui1
    else:
        self.xii_ga = None
        self.emui_ga = None
    comm.Barrier()
    self.win_xii = create_win(self.xii_ga, comm=comm)
    self.win_emui = create_win(self.emui_ga, comm=comm)
    self.win_xii.Fence()
    self.win_emui.Fence()
    

def get_precon_GA(self, pairlist):
    def get_sf_tmp(ij, s_tmp, f_tmp, s_buffer, f_buffer):
        i = ij//self.no
        j = ij%self.no
        ji = j*self.no+i
        ii = i*self.no+i
        jj = j*self.no+j
        s_buffer, s_tmp = read_GA(self.sf_address, [ii, ij, jj], s_buffer, self.win_smat, dtype='f8', list_col=s_tmp, dim_list=self.dim_sf)
        f_buffer, f_tmp = read_GA(self.sf_address, [ii, ij, jj], f_buffer, self.win_fmat, dtype='f8', list_col=f_tmp, dim_list=self.dim_sf)
        if i != j:
            s_tmp[ji] = s_tmp[ij].T
            f_tmp[ji] = f_tmp[ij].T
        return s_tmp, f_tmp
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))[irank]
    #pair_slice = get_pairslice(self, if_remote=False)[irank]
    self.win_xmat_dim, self.dim_xmat = get_shared((self.no**2, 2), dtype='i')
    self.win_emuij_dim, self.dim_emuij = get_shared((self.no**2, 2), dtype='i')
    if pair_slice is not None:
        pair_close = []
        for ipair in pair_slice:
            if self.if_remote[ipair] is False:
                pair_close.append(ipair)
        if pair_close != []:
            dim_xmat = []
            dim_emuij = []
            dim_sf = []
            for ipair in pair_close:
                if self.if_remote[ipair] is False:
                    i = ipair//self.no
                    j = ipair%self.no
                    dim_ij = self.nosv[i] + self.nosv[j]
                    if i == j:
                        dim_ij_remote = self.nosv[i]
                    else:
                        dim_ij_remote = dim_ij
                    dim_xmat.append(dim_ij*dim_ij_remote)
                    dim_emuij.append(dim_ij_remote*dim_ij_remote)
                    if i != j:
                        dim_sf.append(self.nosv[i]**2 + self.nosv[i]*self.nosv[j] + self.nosv[j]**2)
                    else:
                        dim_sf.append(self.nosv[i]**2)
            self.xmat_ga = np.empty(sum(dim_xmat), dtype='f8')
            self.emuij_ga = np.empty(sum(dim_emuij), dtype='f8')
            s_tmp = [None]*self.no**2
            f_tmp = [None]*self.no**2
            s_buffer = np.empty(max(dim_sf), dtype='f8')
            f_buffer = np.copy(s_buffer)
            idx_xmat0, idx_emuij0 = 0, 0
            for idx, ipair in enumerate(pair_close):
                i = ipair//self.no
                j = ipair%self.no
                s_tmp, f_tmp = get_sf_tmp(ipair, s_tmp, f_tmp, s_buffer, f_buffer)
                S_mat = generation_SuperMat([i, j, i, j], s_tmp, self.nosv, self.no)
                #count = 0
                #while True:
                try:
                    eigval, eigvec = eigh(S_mat)
                except np.linalg.linalg.LinAlgError:
                    print_test(S_mat, "S_mat")
                    #get_coords(self.mol)
                    print(get_coords_from_mol(self.mol, coord_only=True))
                    print("numpy.linalg.LinAlgError: Eigenvalues did not converge")
                    subprocess.call(["kill", "-9", "%s"%os.getpid()])
                    '''if count > 5:
                        raise np.linalg.linalg.LinAlgError('Eigenvalues did not converge')
                    else:
                        print('Rank %d: Eigenvalues did not converge, try %d'%(irank, count))
                    count += 1'''

                S_mat = None
                newvec = eigvec[:, eigval>1e-5]/np.sqrt(eigval[eigval>1e-5])
                F_mat = generation_SuperMat([i, j, i, j], f_tmp, self.nosv, self.no)
                newh = multi_dot([newvec.T, F_mat, newvec])
                F_mat = None
                eigval, eigvec = eigh(newh)
                
                eij = self.eo[i]+self.eo[j] 
                eab = eigval+eigval.reshape(-1, 1)

                effective_c = ddot(newvec, eigvec)
                dim_xmat0, dim_xmat1 = effective_c.shape
                self.dim_xmat[ipair] = np.asarray([dim_xmat0, dim_xmat1])
                idx_xmat1 = idx_xmat0 + effective_c.size
                self.xmat_ga[idx_xmat0: idx_xmat1] = effective_c.ravel()

                effective_d = 1.0/(eij - eab)
                dim_emuij0, dim_emuij1 = effective_d.shape
                self.dim_emuij[ipair] = np.asarray([dim_emuij0, dim_emuij1])
                idx_emuij1 = idx_emuij0 + effective_d.size
                self.emuij_ga[idx_emuij0: idx_emuij1] = effective_d.ravel()
                idx_xmat0, idx_emuij0 = idx_xmat1, idx_emuij1
                if ipair != pair_slice[-1]:
                    if i != j:
                        for pair_i in [ipair, j*self.no+j]:
                            s_tmp[pair_i] = None
                            f_tmp[pair_i] = None
                    pair_next = pair_slice[idx+1]
                    i_next = pair_next//self.no
                    if i_next != i:
                        s_tmp[i*self.no+i] = None
                        f_tmp[i*self.no+i] = None
            
            s_tmp, f_tmp = None, None
            s_buffer, f_buffer = None, None
        else:
            self.xmat_ga = None
            self.emuij_ga = None
    else:
        self.xmat_ga = None
        self.emuij_ga = None
    comm.Barrier()
    self.win_xmat = create_win(self.xmat_ga, comm=comm)
    self.win_emuij = create_win(self.emuij_ga, comm=comm)
    Acc_and_get_GA(self.dim_xmat)
    Acc_and_get_GA(self.dim_emuij)
    pair_slice = get_slice(job_list=pairlist, rank_list=range(nrank))
    #pair_slice = get_pairslice(self, if_remote=False)
    self.xmat_address = [None]*self.no**2
    self.emuij_address = [None]*self.no**2
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            idx_xmat0, idx_emuij0 = 0, 0
            for idx, ipair in enumerate(pair_i):
                idx_xmat1 = idx_xmat0 + np.product(self.dim_xmat[ipair])
                self.xmat_address[ipair] = [rank_i, [idx_xmat0, idx_xmat1]]
                idx_emuij1 = idx_emuij0 + np.product(self.dim_emuij[ipair])
                self.emuij_address[ipair] = [rank_i, [idx_emuij0, idx_emuij1]]
                idx_xmat0, idx_emuij0 = idx_xmat1, idx_emuij1
    
def get_ijp_GA(self, mtype="mp2"):
    mo_slice = get_slice(range(nrank), job_list=self.mo_list)
    self.jlist = [None]*self.no
    for ipair in self.pairlist:
        i = ipair//self.no
        j = ipair%self.no
        if self.jlist[i] is None:
            self.jlist[i] = [j]
        else:
            self.jlist[i].append(j)
    self.ijp_address = [None]*self.no**2
    len_ijp_core = 0
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            ijp_idx = 0
            for i in mo_i:
                for j in self.jlist[i]:
                    self.ijp_address[i*self.no+j] = [rank_i, [ijp_idx, ijp_idx+1]]
                    ijp_idx += 1
                    if rank_i == irank:
                        len_ijp_core += 1
    if self.direct_int:
        self.dir_ijp = 'ijp_%s_tmp'%mtype
        if irank == 0:
            make_dir(self.dir_ijp)
        comm.Barrier()
    mo_slice = mo_slice[irank]
    if mo_slice is not None:
        if self.direct_int:
            file_ijp = h5py.File("%s/%d.tmp"%(self.dir_ijp, irank), "w")
            file_ijp.create_dataset('ijp', shape=(len_ijp_core, self.naoaux), dtype='f8')
            buf_ialp = np.empty((self.nao, self.naoaux))
        else:
            self.ijp_ga = np.empty((len_ijp_core, self.naoaux), dtype='f8')
        ijp_idx0 = 0
        for idx_i, i in enumerate(mo_slice):
            ijp_idx1 = ijp_idx0 + len(self.jlist[i])
            if self.direct_int:
                file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                read_file(file_ialp, 'ialp', buffer=buf_ialp)
                ijp_i = np.dot(self.o[:, self.jlist[i]].T, buf_ialp)
                file_ijp['ijp'].write_direct(ijp_i, dest_sel=np.s_[ijp_idx0:ijp_idx1])
            else:
                ialp_tmp = self.ialp_mo[idx_i]
                self.ijp_ga[ijp_idx0:ijp_idx1] = np.dot(self.o[:, self.jlist[i]].T, ialp_tmp)
            ijp_idx0 = ijp_idx1
    else:
        self.ijp_ga = None
    comm.Barrier()
    
    if self.direct_int is False:
        self.win_ijp = create_win(self.ijp_ga, comm=comm)
        self.win_ijp.Fence()
            
def get_imup_GA(self):
    mo_slice = get_slice(range(nrank), job_list=self.mo_list)
    self.dim_imup = [None]*self.no
    self.imup_address = [None]*self.no
    dim_imup = []
    for rank_i, mo_i in enumerate(mo_slice):
        if mo_i is not None:
            idx_imup0 = 0
            for idx, i in enumerate(mo_i):
                self.dim_imup[i] = (self.naoaux, self.nosv[i])
                idx_imup1 = idx_imup0 + self.naoaux*self.nosv[i]
                self.imup_address[i] = [rank_i, [idx_imup0, idx_imup1]]
                if rank_i == irank:
                    dim_imup.append(self.naoaux*self.nosv[i])
                idx_imup0 = idx_imup1
    
    if self.direct_int:
        self.dir_imup = 'imup_tmp'
        if irank == 0:
            make_dir(self.dir_imup)
        comm.Barrier()
    mo_slice = mo_slice[irank]
    if mo_slice is not None:
        if self.direct_int:
            buf_ialp = np.empty((self.nao, self.naoaux))
        else:
            self.imup_ga = np.empty(sum(dim_imup), dtype='f8')
        idx_imup0 = 0
        for idx_i, i in enumerate(mo_slice):
            t_idx0, t_idx1 = self.qmat_address[i][1]
            qmat_tmp = self.qmat_ga[t_idx0: t_idx1].reshape(self.nv, self.nosv[i])

            if self.direct_int:
                file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                read_file(file_ialp, 'ialp', buffer=buf_ialp)
                #imup_tmp = multi_dot([ialp_tmp.T, self.v, qmat_tmp])
                imup_i = multi_dot([buf_ialp.T, self.v, qmat_tmp])
                with h5py.File('%s/imup_%d.tmp'%(self.dir_imup, i), 'w') as file_imup:
                    file_imup.create_dataset('imup', shape=(self.naoaux, self.nosv[i]), dtype='f8')
                    file_imup['imup'].write_direct(imup_i)
            else:
                idx_imup1 = idx_imup0 + self.naoaux*self.nosv[i]
                ialp_tmp = self.ialp_mo[idx_i]
                self.imup_ga[idx_imup0: idx_imup1] = multi_dot([ialp_tmp.T, self.v, qmat_tmp]).ravel()
                #self.imup_ga[idx_imup0: idx_imup1] = multi_dot([ialp_tmp, self.v, qmat_tmp]).ravel()
            idx_imup0 = idx_imup1
    else:
        self.imup_ga = None
    comm.Barrier()
    
    if self.direct_int is False:
        self.win_imup = create_win(self.imup_ga, comm=comm)
        self.win_imup.Fence()
    #read imup for node
    #self.imup_address_node, self.win_imup_node, self.imup_node = get_GA_node(self, self.mos_remote_node, self.win_imup, self.imup_address, self.dim_imup, self.no)


def get_kmatrix_GA(self):
    def sup_pairs(pair_slice):
        def get_maxlen(slice_i):
            max_len = 0
            for si in slice_i:
                if si is not None:
                    if len(si) > max_len:
                        max_len = len(si)
            return max_len
        def get_sup_slice(slice_i, max_len):
            if slice_i is None:
                slice_i = [None]*max_len
            else:
                slice_i = slice_i + [None]*(max_len-len(slice_i))
            return slice_i
        max_len = get_maxlen(pair_slice)
        return get_sup_slice(pair_slice[irank], max_len)

    #For share_node only
    if self.ml_test:
        self.idx_pair = [None]*self.no**2
        self.kmat_address_node = [None]*self.no**2
        idx_k0 = 0
        for pidx, ipair in enumerate(self.refer_pairlist):
            self.idx_pair[ipair] = pidx
            i = ipair//self.no
            j = ipair%self.no
            if (self.if_remote[ipair]):
                idx_k1 = idx_k0 + self.nosv[i]*self.nosv[j]
            else:
                tdim_ij = self.nosv[i]+self.nosv[j]
                idx_k1 = idx_k0 + (tdim_ij)**2
            self.kmat_address_node[ipair] = [idx_k0, idx_k1]
            idx_k0 = idx_k1
        self.win_coulomb_pair, self.coulomb_pair = get_shared(len(self.refer_pairlist))
        self.win_exchange_pair, self.exchange_pair = get_shared(len(self.refer_pairlist))
        self.win_kmat, self.Kmat_osv = get_shared(idx_k1)

    opt = 0
    if opt == 0:
        pair_slice = get_pairslice(self, even_adjust=True)
    else:
        pair_slice = get_pairslice(self, even_adjust=False)
    self.dim_kmat = [None]*self.no**2
    self.kmat_address = [None]*self.no**2
    
    t1 = time_now()
    for rank_i, pair_i in enumerate(pair_slice):
        if pair_i is not None:
            idx_k0 = 0
            for ipair in sorted(pair_i):
                i = ipair//self.no
                j = ipair%self.no
                if (self.if_remote[ipair]):
                    self.dim_kmat[ipair] = (self.nosv[i], self.nosv[j])
                    idx_k1 = idx_k0 + self.nosv[i]*self.nosv[j]
                else:
                    tdim_ij = self.nosv[i]+self.nosv[j]
                    idx_k1 = idx_k0 + (tdim_ij)**2
                    self.dim_kmat[ipair] = (tdim_ij, tdim_ij)
                self.kmat_address[ipair] = [rank_i, [idx_k0, idx_k1]]
                idx_k0 = idx_k1
    #comm.Barrier()
    tt = time_now()
    t_read = np.zeros(2)
    t_cal = np.zeros(2)
    t_syn = np.zeros(2)
    t_cd = np.zeros(2)
    t_solve = np.zeros(2)
    if opt == 0:
        pair_slice = sup_pairs(pair_slice)
    else:
        pair_slice = pair_slice[irank]
    if pair_slice is not None:
        if self.loc_fit:
            from osvmp2.loc.loc_addons import joint_fit_domains_by_atom, joint_fit_domains_by_aux
            fit_ij = [None]*self.no**2
            nfit_ij = [None]*self.no**2
            max_naux = 0
            for ipair in pair_slice:
                if ipair is not None:
                    i = ipair//self.no
                    j = ipair%self.no
                    '''fit_ij[ipair], nfit_ij[ipair] = joint_fit_domains_by_atom(self.with_df.auxmol, [i, j], 
                                                                      self.atom_close, joint_type='union')'''
                    fit_ij[ipair], nfit_ij[ipair] = joint_fit_domains_by_aux(self.with_df.auxmol, [i, j], 
                                                                        self.fit_list, joint_type='union')
                    if max_naux < nfit_ij[ipair]:
                        max_naux = nfit_ij[ipair]
            if pair_slice[0] is not None:
                j2c_buffer = np.empty(max_naux**2)
        dim_kmat = []
        dim_imup = []
        dim_qmat = []
        pair_remote = []
        pair_close = []
        max_osv = 0
        for ipair in pair_slice:
            if ipair is not None:
                i = ipair//self.no
                j = ipair%self.no
                max_osv = max([max_osv, self.nosv[i], self.nosv[j]])
                if (self.if_remote[ipair]):
                    pair_remote.append(ipair)
                    dim_kmat.append(self.nosv[i]*self.nosv[j])
                    '''if self.loc_fit:
                        dim_imup.append(nfit_ij[ipair]*(self.nosv[i]+self.nosv[j]))
                    else:'''
                    dim_imup.append(self.naoaux*(self.nosv[i]+self.nosv[j]))
                else:
                    tdim_ij = self.nosv[i]+self.nosv[j]
                    pair_close.append(ipair)
                    dim_kmat.append(tdim_ij**2)
                    dim_qmat.append(self.nv*(self.nosv[i]+self.nosv[j]))

        if (len(pair_remote)!=0) or (len(pair_close)!=0):
            self.kmat_ga = np.empty(sum(dim_kmat), dtype='f8')
            if self.ml_test:
                ijp_buffer = np.empty((3, self.naoaux), dtype='f8')
                ijp_tmp = [None]*self.no**2
            if len(pair_remote) != 0:
                imup_buffer = np.empty(max(dim_imup), dtype='f8')
                imup_tmp = [None]*self.no
            if len(pair_close) != 0:
                qmat_buffer = np.empty(max(dim_qmat), dtype='f8')
                ialp_buffer = np.empty((2, self.nao, self.naoaux), dtype='f8')
                if self.loc_fit:
                    ialp_cal = np.empty(2*self.nao*max_naux)
        
        
        qmat_tmp = [None]*self.no
        ialp_tmp = [None]*self.no
        idx_k0 = 0
        idx_remote, idx_close = 0, 0
        log = lib.logger.Logger(self.stdout, self.verbose)
        for idx_pair, ipair in enumerate(pair_slice):
            #read nesessary variables
            if ipair is not None:
                i = ipair //self.no
                j = ipair % self.no
                t1 = time_now()
                if i == j:
                    mos_read = [i]
                    pairs_read = [ipair]
                else:
                    mos_read = [i, j]
                    pairs_read = [i*self.no+i, j*self.no+j, ipair]
                if (self.direct_int): 
                    if self.ml_test:
                        for ridx, pair_i in enumerate(pairs_read):
                            core_i, (ijp_idx0, ijp_idx1) = self.ijp_address[pair_i]
                            read_file('%s/%d.tmp'%(self.dir_ijp, core_i), 'ijp', ijp_idx0, ijp_idx1, buffer=ijp_buffer[ridx])
                            ijp_tmp[pair_i] = ijp_buffer[ridx]
                    if (self.if_remote[ipair] is False):
                        qmat_buffer, qmat_tmp = read_GA(self.qmat_address, mos_read, qmat_buffer, self.win_qmat, 
                                                        dtype='f8', list_col=qmat_tmp, dim_list=self.dim_qmat)
                else:
                    if self.ml_test:
                        ijp_buffer, ijp_tmp = read_GA(self.ijp_address, pairs_read, ijp_buffer, self.win_ijp, 
                                                      dtype='f8', list_col=ijp_tmp, sup_dim=self.naoaux)
                    if (self.if_remote[ipair]):
                        imup_buffer, imup_tmp = read_GA(self.imup_address, mos_read, imup_buffer, self.win_imup, 
                                                        dtype='f8', list_col=imup_tmp, dim_list=self.dim_imup)
                    else:
                        ialp_buffer, ialp_tmp = read_GA(self.ialp_mo_address, mos_read, ialp_buffer, self.win_ialp_mo, 
                                                        dtype='f8', list_col=ialp_tmp, sup_dim=(self.nao, self.naoaux))
                        qmat_buffer, qmat_tmp = read_GA(self.qmat_address, mos_read, qmat_buffer, self.win_qmat, 
                                                        dtype='f8', list_col=qmat_tmp, dim_list=self.dim_qmat)
                t_read += time_now() - t1
            if opt == 0:comm.Barrier()
            #Read from files
            if (self.direct_int) and (ipair is not None):
                t1 = time_now()
                if i == j:
                    readlist = [i]
                    idx_ialp = 0
                    idx_imup0 = 0
                else:
                    if (self.if_remote[ipair]):
                        if (imup_tmp[i] is not None):
                            readlist = [j]
                            idx_imup0 = imup_tmp[i].size
                        else:
                            readlist = [i, j]
                            idx_imup0 = 0
                    else:
                        if (ialp_tmp[i] is not None):
                            readlist = [j]
                            idx_ialp = 1    
                        else:
                            readlist = [i, j]
                            idx_ialp = 0
                        
                for k in readlist:
                    if (self.if_remote[ipair]):
                        idx_imup1 = idx_imup0 + self.naoaux*self.nosv[k]
                        imup_tmp[k] = imup_buffer[idx_imup0: idx_imup1].reshape(self.naoaux, self.nosv[k])
                        read_file('%s/imup_%d.tmp'%(self.dir_imup, k), 'imup', buffer=imup_tmp[k])
                        idx_imup0 = idx_imup1
                    else:
                        read_file('%s/ialp_%d.tmp'%(self.dir_ialp, k), 'ialp', buffer=ialp_buffer[idx_ialp])
                        ialp_tmp[k] = ialp_buffer[idx_ialp]
                        idx_ialp += 1
                t_read += time_now() - t1

            if self.ml_test and ipair is not None:
                pidx = self.idx_pair[ipair]
                self.coulomb_pair[pidx] = np.einsum("p,p->", ijp_tmp[i*self.no+i], ijp_tmp[j*self.no+j])
                self.exchange_pair[pidx] = np.einsum("p,p->", ijp_tmp[ipair], ijp_tmp[ipair])
                for pair_i in [i*self.no+i, j*self.no+j, i*self.no+j]:
                    ijp_tmp[pair_i] = None

            if ipair is not None:
                idx_k0, idx_k1 = self.kmat_address[ipair][1]
                if (self.if_remote[ipair]):
                    def slice_imup(i, j):
                        def read_slice(imup_slice, imup_ori):
                            idx_p0 = 0
                            for p0, p1 in fit_ij[ipair]:
                                idx_p1 = idx_p0 + (p1-p0)
                                imup_slice[idx_p0:idx_p1] = imup_ori[p0:p1]
                                idx_p0 = idx_p1
                            return imup_slice
                        #buf_tmp = imup_cal[:2*self.nao*naux].reshape(2, self.nao, naux)
                        imup_i = read_slice(np.empty((naux, self.nosv[i])), imup_tmp[i])
                        imup_j = read_slice(np.empty((naux, self.nosv[j])), imup_tmp[j])
                        return imup_i, imup_j
                    #idx_k1 = idx_k0 + self.nosv[i]*self.nosv[j]
                    t1 = time_now()
                    if self.loc_fit:
                        naux = nfit_ij[ipair]
                        imup_i, imup_j = slice_imup(i, j)
                        self.kmat_ga[idx_k0: idx_k1] = ddot(imup_i.T, imup_j).ravel()
                    else:
                        self.kmat_ga[idx_k0: idx_k1] = ddot(imup_tmp[i].T, imup_tmp[j]).ravel()
                    t_cal += time_now() - t1
                    if ipair != pair_remote[-1]:
                        if i != j:
                            imup_tmp[j] = None
                        pair_next = pair_remote[idx_remote+1]
                        i_next = pair_next//self.no
                        if i_next != i:
                            imup_tmp[i] = None
                    idx_remote += 1
                else:
                    t1 = time_now()
                    vir_coeff = ddot(self.v, np.concatenate((qmat_tmp[i], qmat_tmp[j]), axis=1))
                    t_cal += time_now() - t1
                    if self.loc_fit:
                        naux = nfit_ij[ipair]
                        t1 = time_now()
                        t_cd += time_now() - t1
                    
                        def slice_ialp(i, j):
                            def read_slice(ialp_slice, ialp_ori):
                                idx_p0 = 0
                                for p0, p1 in fit_ij[ipair]:
                                    idx_p1 = idx_p0 + (p1-p0)
                                    ialp_slice[:, idx_p0:idx_p1] = ialp_ori[:, p0:p1]
                                    idx_p0 = idx_p1
                                return ialp_slice
                            if i == j:
                                buf_tmp = ialp_cal[:self.nao*naux].reshape(self.nao, naux)
                                return read_slice(buf_tmp, ialp_tmp[i])
                            else:
                                buf_tmp = ialp_cal[:2*self.nao*naux].reshape(2, self.nao, naux)
                                ialp_i = read_slice(buf_tmp[0], ialp_tmp[i])
                                ialp_j = read_slice(buf_tmp[1], ialp_tmp[j])
                                return ialp_i, ialp_j
                        nosv_ij = self.nosv[i]+self.nosv[j]
                        if i == j:
                            t1 = time_now()
                            ialpi = slice_ialp(i, j)
                            #ialp_i = ialp_j = np.empty((nosv_ij, naux))
                            #ddot(vir_coeff.T, ialpi, out=ialp_i)
                            ialpi = ddot(vir_coeff.T, ialpi)
                            self.kmat_ga[idx_k0: idx_k1] = multi_dot([ialpi, ialpi.T]).ravel()
                            t_cal += time_now() - t1

                        else:
                            t1 = time_now()
                            ialpi, ialpj = slice_ialp(i, j)
                            self.kmat_ga[idx_k0: idx_k1] = multi_dot([vir_coeff.T, ialpi, ialpj.T, vir_coeff]).ravel()
                            t_cal += time_now() - t1
                    else:
                        t1 = time_now()
                        self.kmat_ga[idx_k0: idx_k1] = multi_dot([vir_coeff.T, ialp_tmp[i], ialp_tmp[j].T, vir_coeff]).ravel()
                        t_cal += time_now() - t1
                    if ipair != pair_close[-1]:
                        if i != j:
                            qmat_tmp[j] = None
                            ialp_tmp[j] = None
                        pair_next = pair_close[idx_close+1]
                        i_next = pair_next//self.no
                        if i_next != i:
                            qmat_tmp[i] = None
                            ialp_tmp[i] = None
                    idx_close += 1
                if self.ml_test:
                    idx_kosv0, idx_kosv1 = self.kmat_address_node[ipair]
                    self.Kmat_osv[idx_kosv0: idx_kosv1] = self.kmat_ga[idx_k0: idx_k1]
        if pair_slice[0] is None:
            self.kmat_ga = None
        
    else:
        self.kmat_ga = None
    
    
    log = lib.logger.Logger(self.stdout, self.verbose)
    time_list = [['reading', t_read], ['transformation', t_cal]]
    '''if self.loc_fit:
        time_list += [['cholesky decomp.', t_cd], ['solve tri', t_solve]]'''
    print_time(time_list, log)
    comm.Barrier()
    #print_time(['kmat', time_elapsed(tt)], log); sys.exit()
    
    self.win_kmat = create_win(self.kmat_ga, comm=comm)
    self.win_kmat.Fence()
    if (self.grad_cal is False) and (self.direct_int is False):
        free_win(self.win_ialp_mo)
        self.ialp_aux = None
        self.ialp_mo = None

def get_mp2e_GA(self):
    def get_residue(ipair, K_matrix, S_matrix, F_matrix, T_matrix, k_list=None):
        R_matrix = np.copy(K_matrix[ipair])
        i = ipair//self.no
        j = ipair % self.no
        if self.if_remote[ipair]:
            T_ii = T_matrix[i*self.no+i][:self.nosv[i], self.nosv[i]:]
            T_jj = T_matrix[j*self.no+j][:self.nosv[j], self.nosv[j]:]
            R_matrix += ddot(T_matrix[ipair], (F_matrix[j*self.no+j] - self.loc_fock[j,j]))
            R_matrix += ddot((F_matrix[i*self.no+i] - self.loc_fock[i,i]), T_matrix[ipair])
            R_matrix -= self.loc_fock[i,j] * (ddot(S_matrix[i*self.no+j], T_jj) + 
                                              ddot(T_ii, S_matrix[i*self.no+j]))
        else:
            k_tol = 1e-5
            F_ijij = generation_SuperMat([i, j, i, j], F_matrix, self.nosv, self.no)
            for k in k_list:
                if abs(self.loc_fock[k, j]) > k_tol:
                    S_ikij = generation_SuperMat([i, k, i, j], S_matrix, self.nosv, self.no)
                    B = -self.loc_fock[k, j] * S_ikij
                    if (k==j):
                        B += F_ijij
                    if i > k:
                        T_ik = flip_ij(k, i, T_matrix[k*self.no+i], self.nosv)
                    else:
                        T_ik = T_matrix[i*self.no+k]
                    try:
                        R_matrix += multi_dot([S_ikij.T, T_ik, B])
                    except ValueError:
                        print(R_matrix.shape, K_matrix[ipair], S_ikij.shape, T_ik.shape)
                if abs(self.loc_fock[i, k]) > k_tol:
                    S_ijkj = generation_SuperMat([i, j, k, j], S_matrix, self.nosv, self.no)
                    C = -self.loc_fock[i, k] * S_ijkj
                    if (i==k):
                        C += F_ijij
                    if k > j:
                        T_kj = flip_ij(j, k, T_matrix[j*self.no+k], self.nosv)
                    else:
                        T_kj = T_matrix[k*self.no+j]
                    R_matrix += multi_dot([C, T_kj, S_ijkj.T])
        return R_matrix
        
    def ene_iteration(pairlist, pair_type, Tmat_save, ene_list, ene_tol, use_diis=False):
        def get_pair(i, j):
            if i < j:
                return i*self.no+j
            else:
                return j*self.no+i
        def sup_pairs(pair_slice):
            def get_maxlen(slice_i):
                max_len = 0
                for si in slice_i:
                    if si is not None:
                        if len(si) > max_len:
                            max_len = len(si)
                return max_len
            def get_sup_slice(slice_i, max_len):
                if slice_i is None:
                    slice_i = [None]*max_len
                else:
                    slice_i = slice_i + [None]*(max_len-len(slice_i))
                return slice_i
            max_len = get_maxlen(pair_slice)
            return get_sup_slice(pair_slice[irank], max_len)
        
        def batch_read(dim_list, addr_list, win_i, pairs):
            t0 = time_now()
            buf = np.empty(get_buff_size(dim_list, pairs))
            buf_list = [None]*len(dim_list)
            buf, buf_list = read_GA(addr_list, pairs, buf, win_i, dtype='f8', 
                                    list_col=buf_list, dim_list=dim_list)
            self.t_read_res += time_now() - t0
            return buf_list
        def get_pairs_mat(pairs):
            pairs_smat = []
            pairs_fmat = []
            pairs_tmat = []
            for ipair in pairs:
                i = ipair//self.no
                j = ipair%self.no
                if self.if_remote[i*self.no+j]:
                    pairs_smat.extend([i*self.no+j])
                    pairs_fmat.extend([i*self.no+i, j*self.no+j])
                    pairs_tmat.extend([i*self.no+i, i*self.no+j, j*self.no+j])
                else:
                    pairs_fmat.extend([i*self.no+i, i*self.no+j, j*self.no+j])
                    pairs_smat.extend([i*self.no+i, i*self.no+j, j*self.no+j])
                    pairs_tmat.extend([i*self.no+j])
                    for k in self.k_list[ipair]:
                        ik, kj = get_pair(i,k), get_pair(k,j)
                        pairs_smat.extend([ik, kj])
                        pairs_tmat.extend([ik, kj])
            return sorted(set(pairs_smat)), sorted(set(pairs_fmat)), sorted(set(pairs_tmat))
        def fill_ji(mat_list, pairs):
            for ipair in pairs:
                i = ipair//self.no
                j = ipair % self.no
                if i != j:
                    mat_list[j*self.no+i] = mat_list[ipair].T
            return mat_list
        def read_mat1(ipair, t_only=False):
            #Read matrices required
            pairs_smat, pairs_fmat, pairs_tmat = get_pairs_mat([ipair])
            T_matrix = batch_read(self.dim_tmat, self.tmat_address, self.win_tmat, pairs_tmat)
            if t_only:
                return T_matrix
            else:
                S_matrix = batch_read(self.dim_sf, self.sf_address, self.win_smat, pairs_smat)
                F_matrix = batch_read(self.dim_sf, self.sf_address, self.win_fmat, pairs_fmat)
                S_matrix = fill_ji(S_matrix, pairs_smat)
                F_matrix = fill_ji(F_matrix, pairs_fmat)
                K_matrix = batch_read(self.dim_kmat, self.kmat_address, self.win_kmat, [ipair])
                return S_matrix, F_matrix, T_matrix, K_matrix
        def read_mat2(ipair, read_k=True):
            if read_k:
                K_matrix = batch_read(self.dim_kmat, self.kmat_address, self.win_kmat, [ipair])
            if self.if_remote[ipair]:
                i = ipair//self.no
                j = ipair%self.no
                emu_i = batch_read(self.dim_emui, self.emui_address, self.win_emui, [i, j])
                x_ii = batch_read(self.dim_xii, self.xii_address, self.win_xii, [i, j])
                if read_k:
                    return emu_i, x_ii, K_matrix
                else:
                    return emu_i, x_ii
            else:
                emu_ij = batch_read(self.dim_emuij, self.emuij_address, self.win_emuij, [ipair])
                X_matrix = batch_read(self.dim_xmat, self.xmat_address, self.win_xmat, [ipair])
                if read_k:
                    return emu_ij, X_matrix, K_matrix
                else:
                    return emu_ij, X_matrix
        def get_mo(pairs):
            mo_list = []
            for ipair in pairs:
                mo_list.extend([ipair//self.no, ipair%self.no])
            return sorted(set(mo_list))

        def compute_ene(ipair, emu_ij, X_matrix, R_matrix, K_matrix, Tmat_save, ene_list):
            if self.if_remote[ipair]:
                i = ipair//self.no
                j = ipair%self.no
                eff_del = 1/(self.eo[i]+self.eo[j]-emu_ij[i].reshape(-1,1)-emu_ij[j].ravel())
                effective_R = eff_del * multi_dot([X_matrix[i].T, R_matrix, X_matrix[j]])
                delta = multi_dot([X_matrix[i], effective_R, X_matrix[j].T])
                Tmat_save[ipair] += delta
                ene_i = 2*ddot(K_matrix[ipair].ravel(), Tmat_save[ipair].ravel())
                if self.ml_test:
                    self.ene_decom_nonmbe[self.idx_pair[ipair], 2] = 2 * ene_i
            else:
                effective_R = emu_ij[ipair] * multi_dot([X_matrix[ipair].T, R_matrix, X_matrix[ipair]])
                delta = multi_dot([X_matrix[ipair], effective_R, X_matrix[ipair].T])
                Tmat_save[ipair] += delta
                T_bar_i = (2*Tmat_save[ipair] - Tmat_save[ipair].T).ravel()
            #compute energies
                ene_i = ddot(K_matrix[ipair].ravel(), T_bar_i)
                '''if self.ml_test:
                    #Energy decomposition
                    def split_block(i, j, mat, pos='tr'):
                        if pos == 'tl':
                            return mat[:self.nosv[i], :self.nosv[i]]
                        elif pos == 'tr':
                            return mat[:self.nosv[i], self.nosv[i]:]
                        elif pos == 'bl':
                            return mat[self.nosv[i]:, :self.nosv[i]]
                        elif pos == 'br':
                            return mat[self.nosv[i]:, self.nosv[i]:]
                        else:
                            raise IndexError("Please specify the block positions with [tl, tr, bl, br]")
                    i = ipair//self.no
                    j = ipair%self.no
                    if i == j:
                        self.ene_decom_nonmbe[self.idx_pair[ipair], 0] = ene_i
                    else:
                        tii = split_block(i, j, Tmat_save[ipair], pos='tl')
                        tjj = split_block(i, j, Tmat_save[ipair], pos='br')
                        tij = split_block(i, j, Tmat_save[ipair], pos='tr')
                        tji = split_block(i, j, Tmat_save[ipair], pos='bl')
                        eii = ddot(split_block(i, j, K_matrix[ipair], 'tl').ravel(), (2*tii-tii.T).ravel())
                        ejj = ddot(split_block(i, j, K_matrix[ipair], 'br').ravel(), (2*tjj-tjj.T).ravel())
                        eij = ddot(split_block(i, j, K_matrix[ipair], 'tr').ravel(), (2*tij-tji.T).ravel())
                        eji = ddot(split_block(i, j, K_matrix[ipair], 'bl').ravel(), (2*tji-tij.T).ravel())
                        
                        tii = split_block(i, j, Tmat_save[ipair], pos='tl')
                        tjj = split_block(i, j, Tmat_save[ipair], pos='br')
                        tij = split_block(i, j, Tmat_save[ipair], pos='tr')
                        tji = split_block(i, j, Tmat_save[ipair], pos='bl')
                        eii = ddot(split_block(i, j, K_matrix[ipair], 'tl').ravel(), (2*tii-tii.T).ravel())
                        ejj = ddot(split_block(i, j, K_matrix[ipair], 'br').ravel(), (2*tjj-tjj.T).ravel())
                        kij = split_block(i, j, K_matrix[ipair], 'tr')
                        kji = split_block(i, j, K_matrix[ipair], 'bl')
                        eij = 2 * ddot(kij.ravel(), tij.ravel()) - ddot(kji.ravel(), tij.T.ravel())
                        eji = 2 * ddot(kji.ravel(), tji.ravel()) - ddot(kij.ravel(), tji.T.ravel())
                        
                        
                        tii = tjj = tij = tji = kij = kji = None
                        self.ene_decom_nonmbe[self.idx_pair[ipair], 1] = 2 * (eii + ejj)
                        self.ene_decom_nonmbe[self.idx_pair[ipair], 2] = 2 * eij
                        self.ene_decom_nonmbe[self.idx_pair[ipair], 3] = 2 * eji'''
            '''if self.mbe_mode == 0:
                ene_i += ddot(R_matrix.ravel(), T_bar_i)'''
            i = ipair//self.no
            j = ipair%self.no
            if i != j:
                ene_i *= 2
            ene_list[ipair] = ene_i
            return Tmat_save, ene_list
        ite = 0
        ene_old = 0.0
        converge = False
        log.info("    Residual iteration for %s pairs"%pair_type)
        read_mode = 0
        t0 = time_now()
        if pair_type == 'remote':
            pair_slice = self.pairslice_remote_tmat
        else:
            pair_slice = self.pairslice_close_tmat
            nk_pairs = np.asarray([0]*self.no**2)
            for ipair in pairlist:
                nk_pairs[ipair] = len(self.k_list[ipair])
            log.info('    The average k loops: %d'%(np.mean(nk_pairs[pairlist])))
        self.t_read_res += time_now() - t0
        r_mode = 0
        if pair_type == 'remote':
            r_mode = 0
        #comm.Barrier()
        pair_slice = sup_pairs(pair_slice)
        while (not converge):
            self.ite = ite
            ene_new = 0.0
            #if pair_slice is not None:
            if r_mode == 1:
                rmat_save = [None]*self.no**2
            for ipair in pair_slice:
                if ipair is not None:
                    S_matrix, F_matrix, T_matrix, K_matrix = read_mat1(ipair)
                    if r_mode == 0:
                        emu_ij, X_matrix = read_mat2(ipair, read_k=False)
                comm.Barrier()
                if ipair is not None:
                    R_matrix = get_residue(ipair, K_matrix, S_matrix, F_matrix, T_matrix, self.k_list[ipair])
                    if r_mode == 1:
                        rmat_save[ipair] = R_matrix
                    else:
                        Tmat_save, ene_list = compute_ene(ipair, emu_ij, X_matrix, R_matrix, K_matrix, Tmat_save, ene_list)
            if r_mode == 1:
                comm.Barrier()
                for ipair in pair_slice:
                    if ipair is not None:
                        emu_ij, X_matrix, K_matrix = read_mat2(ipair)
                        Tmat_save, ene_list = compute_ene(ipair, emu_ij, X_matrix, rmat_save[ipair], K_matrix, Tmat_save, ene_list)
            comm.Barrier()
            Acc_and_get_GA(var=ene_list)
            comm.Barrier()
            ene_new = np.sum(ene_list[pairlist])
            var = abs(ene_old - ene_new)
            ene_old = ene_new
            log.info('    Iter. %d: energy %.10f, by energy increment %.2E', ite, ene_new, Decimal(var))   
            #converged or not
            if (var < ene_tol) or (pair_type == "remote") or (self.local_type==0) or (self.mbe_mode==0):
                converge = True
            else:
                ite += 1            
            if(ite > self.max_cycle): 
                log.warn('OSV-MP2 exceeds the maximum iteration number %d and will quit!', self.max_cycle)
                converge = True
            if (converge):
                return ene_list, ene_new, Tmat_save
            else:
                comm_shm.Barrier()
                if irank_shm == 0:
                    ene_list[:] = 0.0
                comm_shm.Barrier()
            
    log = lib.logger.Logger(self.stdout, self.verbose)
    self.t_read_res = np.zeros(2)
    win_ene, ene_list = get_shared(self.no**2)
    Tmat_save = [None]*self.no**2
    pair_slice = self.pairslice_all_tmat[irank]
    if pair_slice is not None:
        for ipair in pair_slice:
            idx0, idx1 = self.tmat_address[ipair][1]
            Tmat_save[ipair] = self.tmat_ga[idx0:idx1].reshape(self.dim_tmat[ipair])
    ene_list, ene_close, Tmat_save = ene_iteration(self.refer_pairlist_close, "close", Tmat_save, ene_list, self.ene_tol)
    msg_list = [['OSV-MP2 energy converged!', ''], ['Correlation energy of close pairs:', '%.10f'%ene_close]]
    if (len(self.refer_pairlist_remote)>0):
        ene_list, ene_remote, Tmat_save = ene_iteration(self.refer_pairlist_remote, "remote", Tmat_save, ene_list, self.ene_tol)
        msg_list.append(['Correlation energy of remote pairs:', '%.10f'%ene_remote])
    else:
        ene_remote = 0.0
    if ("ene" in self.cal_mode) and (self.save_pene) and (irank == 0):
        elist = [ene_list[ipair] for ipair in self.refer_pairlist] 
        file_pene = "pair_energy.hdf5"
        if os.path.isfile(file_pene):
            file_mode = "r+"
        else:
            file_mode = "w"
        with h5py.File(file_pene, file_mode) as f:
            if self.mol_name not in f.keys():
                f["%s/pairlist"%self.mol_name] = self.refer_pairlist
                f["%s/pair_ene"%self.mol_name] = elist
    ene_total = (ene_close+ene_remote)
    msg_list.extend([['', '-'*len(list('%.10f'%ene_total))], ['MP2 correlation energy:', '%.10f'%ene_total]])
    print_align(msg_list, align='lr', indent=4, log=log)
    free_win(win_ene)
    print_time(['syn. and reading', self.t_read_res], log)
    return ene_close, ene_remote



def get_gamma_GA(self):
    #if self.direct_int:
    def update_cpn(idx, i, N):
        for m in range(self.nosv_cp[i]):
            for n in range(self.nosv[i]):
                delta = self.s_ga[idx][n] - self.s_ga[idx][m]
                #if (abs(delta) <= 1e-6):
                #if m == n:
                #if m <= n:
                #if (abs(delta) <= 1e-8):
                if m == n:
                    N[m, n] = np.float64(0)
                else:
                    if (abs(N[m, n]/delta) > 1):
                        N[m, n] = np.float64(0)
                    else:
                        N[m, n] = N[m, n]/delta
        '''scp_i = self.s_ga[idx, :self.nosv_cp[i]]
        s_i = self.s_ga[idx, :self.nosv[i]]
        delta = lib.direct_sum('a-b->ab', scp_i, s_i).ravel()
        delta *= -1
        idx_large = abs(delta) > 1e-6
        delta[idx_large] = 1/delta[idx_large]
        delta[np.invert(idx_large)] = 0
        N *= delta.reshape(N.shape)'''
        return N
    #Yi -> Yli, Yla, gamma_omug, PQ
    def preread(self):
        mo_slice = get_slice(range(nrank), job_list=self.mo_list)
        node_now = irank//self.nrank_shm
        rank_list = [node_now*comm_shm.size+irank_shm for irank_shm in range(comm_shm.size)]
        mo_slice_node = []
        for rank_i in rank_list:
            if mo_slice[rank_i] is not None:
                mo_slice_node.extend(mo_slice[rank_i])
        #j_list = []
        j_close = []
        j_remote = []
        pairs_node = []
        pairs_close = []
        pairs_remote = []
        for i in mo_slice_node:
            if i is not None:
                for j in self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    if self.if_discarded[ipair] is False:
                        #j_list.append(j)
                        pairs_node.append(ipair)
                        if self.if_remote[ipair]:
                            j_remote.append(j)
                            pairs_remote.append(ipair)
                        else:
                            j_close.append(j)
                            pairs_close.append(ipair)
        j_close = list(set(j_close))
        j_remote = list(set(j_remote))
        pairs_node = sorted(list(set(pairs_node)))
        self.win_imup_node, self.win_qmat_node, self.win_tbar_node = None, None, None
        if (len(j_remote) != 0) and (self.direct_int is False):
            self.imup_address_node, self.win_imup_node, self.imup_node = get_GA_node(self, j_remote, self.win_imup, self.imup_address, self.dim_imup, self.no)
        if len(j_close) != 0:
            self.qmat_address_node, self.win_qmat_node, self.qmat_node = get_GA_node(self, j_close, self.win_qmat, self.qmat_address, self.dim_qmat, self.no)
        #self.tbar_address_node, self.win_tbar_node, self.tbar_node = get_GA_node(self, self.pairs_node, self.win_tbar, self.tbar_address, self.dim_tbar, self.no**2)
        self.tmat_address_node, self.win_tmat_node, self.tmat_node = get_GA_node(self, pairs_node, self.win_tmat, self.tmat_address, self.dim_tmat, self.no**2)
        comm.Barrier()
        
    def clear_preread(self):
        for win_i in [self.win_imup_node, self.win_qmat_node, self.win_tmat_node]:
            if win_i is not None:
                free_win(win_i)
        self.imup_node, self.qmat_node, self.tmat_node = None, None, None
    def transform_yi(self):
        from osvmp2.loc.loc_addons import slice_fit, joint_fit_domains_by_atom, joint_fit_domains_by_aux
        #self.loc_fit = False
        if self.direct_int:
            self.dir_yi = 'yi_tmp'
            self.dir_cal = '%s/cal'%self.dir_yi
            if irank == 0:
                for dir_i in [self.dir_yi, self.dir_cal]:#, self.dir_pq]:
                    make_dir(dir_i)
            comm.Barrier()
        preread(self)
        mo_slice = get_slice(range(nrank), job_list=self.mo_list)
        mo_address = [None]*(self.no+1)
        for rank_i, mo_i in enumerate(mo_slice):
            if mo_i is not None:
                for idx, num in enumerate(mo_i):
                    mo_address[num] = [rank_i, [idx, idx+1]]
        mo_slice = mo_slice[irank]
        win_lqa, Lqa_node = get_shared((self.nao, self.nao), dtype='f8')
        win_pq, pq_node = get_shared((naoaux, naoaux), dtype='f8')
        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            if os.path.isfile('j2c_mp2.tmp'):
                read_file('j2c_mp2.tmp', 'low', buffer=low_node)
            else:
                j2c = self.auxmol.intor('int2c2e', hermi=1)
                low_node[:] = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
        comm_shm.Barrier()

        if mo_slice is not None:
            if irank_shm == 0:
                lqa_tmp = Lqa_node
                buf_pq = pq_node
            else:
                lqa_tmp = np.zeros((self.nao, self.nao))
                buf_pq = np.zeros((naoaux, naoaux))
            mo0, mo1 = mo_slice[0], mo_slice[-1]+1
            nocc_rank = len(mo_slice)
            #Set up buffers
            dim_qmat = []
            dim_tbar = []
            dim_imup = []
            if self.loc_fit:
                fit_ij = [None]*self.no**2
                nfit_ij = [None]*self.no**2
                dim_ialp_lfit = []
                dim_imup_lfit = []
                dim_pjij_lfit = []
                bfit_ij = [None]*self.no**2
                nbfit_ij = [None]*self.no**2
                dim_ialp_lbfit = []
                dim_imup_lbfit = []
                dim_pjij_lbfit = []

                for i in mo_slice:
                    for j in self.mo_list:
                        if i < j:
                            ipair = i*self.no+j
                        else:
                            ipair = j*self.no+i
                        if (self.if_discarded[ipair]): continue
                        if self.loc_fit:
                            '''fit_ij[ipair], nfit_ij[ipair] = joint_fit_domains_by_atom(self.with_df.auxmol, [i, j], 
                                                                            self.atom_close, joint_type='union')'''
                            fit_ij[ipair], nfit_ij[ipair] = joint_fit_domains_by_aux(self.with_df.auxmol, [i, j], 
                                                                            self.fit_list, joint_type='union')
                            bfit_ij[ipair], nbfit_ij[ipair] = joint_fit_domains_by_atom(self.with_df.auxmol, [i, j], 
                                                                            self.atom_close, joint_type='union')
                        if (self.if_remote[ipair]):
                            dim_imup.append(np.product(self.dim_imup[j]))
                            '''if self.loc_fit:
                                dim_imup_lfit.append((self.nosv[i]+self.nosv[j])*nfit_ij[ipair])'''
                        elif self.loc_fit:
                            dim_ialp_lfit.append(2*self.nao*nfit_ij[ipair])
                            dim_pjij_lfit.append((self.nosv[i]+self.nosv[j])*nfit_ij[ipair])
                            dim_ialp_lbfit.append(2*self.nao*nbfit_ij[ipair])
                            dim_pjij_lbfit.append((self.nosv[i]+self.nosv[j])*nbfit_ij[ipair])
            if len(dim_imup) != 0:
                buf_imup = np.empty(max(dim_imup))
            buf_ti = np.empty((self.nv, self.nv))
            if self.direct_int:
                yi = np.empty((self.nao, naoaux))
                buf_qcp = np.empty(self.nv*max(self.nosv_cp[mo_slice]))
                buf_ialp = np.empty((2, self.nao, naoaux))
                ialp_i = buf_ialp[0]
                ialp_j = buf_ialp[1]
                if self.loc_fit:
                    buf_ialp_lfit = np.empty(max(dim_ialp_lfit))
                    buf_pjij_lfit = np.empty(max(dim_pjij_lfit))
                with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, irank), 'w') as file_yi:
                    #file_yi.create_dataset('yi', shape=(nocc_rank, naoaux, self.nao), dtype='f8')
                    file_yi.create_dataset('yi', shape=(nocc_rank, self.nao, naoaux), dtype='f8')
            else:
                self.yi_mo = np.zeros((nocc_rank, self.nao, naoaux))
                #ialp_j = np.empty((self.nao, naoaux), dtype='f8')
                buf_ialp = np.empty((self.nao, naoaux), dtype='f8')
                ialp_j = buf_ialp
 
            if irank_shm == 0:
                DMP2 = self.DMP2
            else:
                DMP2 = self.dmp2_save
            for idx_i, i in enumerate(mo_slice):
                t0 = time_now()
                #Read ialp_i and Q'_i
                if self.direct_int:
                    file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                    read_file(file_ialp, 'ialp', buffer=ialp_i)

                    file_qcp = '%s/qcp_%d.tmp'%(self.dir_qcp, i)
                    qcp_i = buf_qcp[:self.nv*self.nosv_cp[i]].reshape(self.nv, self.nosv_cp[i])
                    read_file(file_qcp, 'qcp', buffer=qcp_i)
                    yi[:] = 0.0
                else:
                    rank_i, (idx0, idx1) = self.qcp_address[i]
                    qcp_i = self.qcp_ga[idx0:idx1].reshape(self.nv, self.nosv_cp[i])
                    ialp_i = self.ialp_mo[idx_i]
                    yi = self.yi_mo[idx_i]
                #Read qmat_i
                rank_i, (idx0, idx1) = self.qmat_address[i]
                qmat_i = self.qmat_ga[idx0:idx1].reshape(self.dim_qmat[i])

                #Initialse imu0 and pjij
                j_list = []
                j_remote = []
                for j in self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    if self.if_discarded[ipair] is False:
                        j_list.append(j)
                        if self.if_remote[ipair]:
                            j_remote.append(j)
                
                if len(j_remote) != 0:
                    pjij = np.zeros((naoaux, self.nosv[i]))
                if (self.use_cposv):
                    imu0 = np.zeros((self.nao, self.nosv[i]))
                    #Read N_i
                    n_idx0, n_idx1 = self.cpn_address[i][1]
                    N_i = self.cpn_ga[n_idx0: n_idx1].reshape(self.dim_cpn[i])
                    self.t_read += time_now() - t0
                
                tj = time_now()
                for j in j_list:#self.mo_list:
                    if i < j:
                        ipair = i*self.no+j
                    else:
                        ipair = j*self.no+i
                    #if self.if_discarded[ipair]: continue
                    
                    t0 = time_now()
                    #Read T_bar
                    tmat_tmp = read_GA_node([ipair], self.tmat_address_node, self.tmat_node, self.dim_tmat)

                    self.t_read += time_now() - t0
                    if self.if_remote[ipair]:
                        t1 = time_now()
                        t0 = time_now()
                        #Read J_j Q_j
                        if self.direct_int:
                            imup_tmp = buf_imup[:naoaux*self.nosv[j]].reshape(naoaux, self.nosv[j])
                            read_file('%s/imup_%d.tmp'%(self.dir_imup, j), 'imup', buffer=imup_tmp)
                        else:
                            imup_tmp = read_GA_node([j], self.imup_address_node, self.imup_node, self.dim_imup)
                        self.t_read += time_now() - t0

                        t0 = time_now()
                        tbar_tmp = 2*4*tmat_tmp
                        if i < j:
                            pjij += ddot(imup_tmp, tbar_tmp.T)
                        else:
                            pjij += ddot(imup_tmp, tbar_tmp)
                        self.t_dk += time_now() - t0
                        self.t_remote += time_now() - t1
                    else:
                        t1 = time_now()
                        if i != j:
                            t0 = time_now()
                            #Read qmat_j
                            qmat_j = read_GA_node([j], self.qmat_address_node, self.qmat_node, self.dim_qmat)

                            #Read ialp_j
                            if self.direct_int:
                                file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, j)
                                read_file(file_ialp, 'ialp', buffer=ialp_j)                                    
                            else:
                                rank_i, idx_list = self.ialp_mo_address[j]
                                idx0, idx1 = idx_list
                                sup_dim = self.nao*naoaux
                                target = [idx0*sup_dim*8, (idx1-idx0)*sup_dim, MPI.DOUBLE]
                                Get_GA(self.win_ialp_mo, ialp_j, target_rank=rank_i, target=target)
                            self.t_read += time_now() - t0
                        else:
                            qmat_j = qmat_i
                            ialp_j = ialp_i
                        

                        t0 = time_now()
                        #Compute first term of Y_i
                        if i < j:
                            coeff = ddot(self.v, np.concatenate((qmat_i, qmat_j), axis=1))
                        else:
                            coeff = ddot(self.v, np.concatenate((qmat_j, qmat_i), axis=1))
                        coeff *= 4
                        #Get J_j^dag(Q_i Q_j)
                        tbar_tmp = 2*tmat_tmp - tmat_tmp.T
                        if i < j:
                            pj_ij_full = multi_dot([ialp_j.T, coeff, tbar_tmp.T])
                        else:
                            pj_ij_full = multi_dot([ialp_j.T, coeff, tbar_tmp])
                        #Compute first term of Y_i
                        yi += ddot(coeff/4, pj_ij_full.T) 
                        self.t_yi += time_now() - t0

                        if (self.use_cposv):
                            #Compute derivative K
                            def slice_lfit(ialp_i, ialp_j, pj_ij_full):
                                t0 = time_now()
                                #Slice fitting domains of ialp
                                ialp_tmp = buf_ialp_lfit[:2*self.nao*naux].reshape(2, self.nao, naux)
                                ialp_i = slice_fit(ialp_tmp[0], ialp_i, fit_ij[ipair], axis=1)
                                if i == j:
                                    ialp_j = ialp_i
                                else:
                                    ialp_j = slice_fit(ialp_tmp[1], ialp_j, fit_ij[ipair], axis=1)
                                pj_ij_tmp = buf_pjij_lfit[:naux*(self.nosv[i]+self.nosv[j])].reshape(naux, (self.nosv[i]+self.nosv[j]))
                                pj_ij_tmp = slice_fit(pj_ij_tmp, pj_ij_full, fit_ij[ipair], axis=0)
                                self.t_slice += time_now() - t0
                                return ialp_i, ialp_j, pj_ij_tmp
                            t0 = time_now()
                            if self.loc_fit:
                                naux = nfit_ij[ipair]
                            else:
                                naux = self.naoaux
                            if (self.loc_fit is False) or (naux==self.naoaux):
                                pj_ij = pj_ij_full
                            else:
                                ialp_i, ialp_j, pj_ij = slice_lfit(ialp_i, ialp_j, pj_ij_full)
                            if i < j:
                                imu0 += ddot(ialp_i, pj_ij[:, :self.nosv[i]])
                                imu0 += multi_dot([ialp_j, ialp_i.T, coeff, tbar_tmp[:, :self.nosv[i]]])
                            else:
                                if i == j:
                                    imu0 += 2*ddot(ialp_i, pj_ij[:, self.nosv[j]:])
                                else:
                                    imu0 += ddot(ialp_i, pj_ij[:, self.nosv[j]:])
                                    imu0 += multi_dot([ialp_j, ialp_i.T, coeff, tbar_tmp.T[:, self.nosv[j]:]])
                            self.t_dk += time_now() - t0
                        
                        if self.loc_fit:
                            ialp_i = buf_ialp[0]
                            ialp_j = buf_ialp[1]
                        elif i == j:
                            if self.direct_int:
                                ialp_j = buf_ialp[1]
                            else:
                                ialp_j = buf_ialp
                self.t_jloop += time_now() - tj
                
                #ti = time_now()
                if j_remote != []:
                    t0 = time_now()
                    yi += multi_dot([self.v, qmat_i, pjij.T])
                    self.t_yi += time_now() - t0
                if (self.use_cposv):
                    t0 = time_now()
                    N_i += multi_dot([qcp_i.T, self.v.T, imu0])
                    self.t_dk += time_now() - t0
                    self.t_close += time_now() - t1
                    if j_remote != []:
                        t0 = time_now()
                        N_i += multi_dot([qcp_i.T, self.v.T, ialp_i, pjij])
                        self.t_dk += time_now() - t0

                    #Read T_ii
                    t0 = time_now()
                    file_ti = '%s/ti_%d.tmp'%(self.dir_ti, i)
                    read_file(file_ti, 'ti', buffer=buf_ti)
                    self.t_read += time_now() - t0

                    t0 = time_now()
                    N_i = update_cpn(idx_i, i, N_i)
                    eiab = lib.numpy_helper.direct_sum('a+b->ab', self.ev, self.ev)-2*self.eo[i]
                    omega = multi_dot([qcp_i, N_i, qmat_i.T])/eiab
                    xi = multi_dot([self.v, omega, self.v.T])
                    xi += xi.T
                    yi += ddot(xi, ialp_i)
                    #omega += omega.T
                    omega *= buf_ti
                    '''omega *= 2
                    DMP2[i, i] += np.sum(omega)
                    DMP2[self.no:, self.no:] -= omega'''
                    
                    DMP2[i, i] += 2*np.sum(omega)
                    DMP2[self.no:, self.no:] -= np.diag(np.sum(omega, axis=0) + 
                                                        np.sum(omega, axis=1))
                    omega = None
                else:
                    file_ti = None

                ti = time_now()
                #compute y_al be
                lqa_tmp += ddot(ialp_i, yi.T)

                #compute PQ
                buf_pq += ddot(ialp_i.T, yi)

                scipy.linalg.solve_triangular(low_node.T, yi.T, lower=False, overwrite_b=True, 
                                              check_finite=False)
                self.t_yi += time_now() - t0
                self.t_iloop += time_now() - ti
                if self.direct_int:
                    for f in [file_ti, file_qcp]:
                        if f is not None:
                            os.remove(f)
                    t1 = time_now()
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, irank), 'r+') as file_yi:
                        #file_yi['yi'][idx_i] = yi.T
                        file_yi['yi'].write_direct(yi, dest_sel=np.s_[idx_i])
                    self.t_write += time_now() - t1
                    yi = yi.reshape(self.nao, naoaux)
            if irank_shm != 0:
                Accumulate_GA_shm(win_lqa, Lqa_node, lqa_tmp)
                Accumulate_GA_shm(win_pq, pq_node, buf_pq)
                lqa_tmp, buf_pq= None, None
        else:
            buf_pq = None
        #win_pq.Fence()
        if irank_shm != 0:
            Accumulate_GA_shm(self.win_dmp2, self.DMP2, self.dmp2_save)
        comm.Barrier()
        clear_preread(self)
        Acc_and_get_GA(var=self.DMP2)
        Acc_and_get_GA(Lqa_node)
        Accumulate_GA(var=pq_node)

        if irank_shm == 0:
            self.Yla = multi_dot([self.mo_coeff.T, Lqa_node, self.RHF.get_ovlp(), self.v])
        else:
            self.Yla = None
        comm.Barrier()
        free_win(win_lqa); Lqa_node=None; lqa_tmp=None

        t0 = time_now()
        if irank == 0:
            if self.direct_int:
                shutil.rmtree(self.dir_ialp)
            buf_pq *= 0.5
            scipy.linalg.solve_triangular(low_node.T, buf_pq.T, lower=False, overwrite_b=True, 
                                          check_finite=False)
            buf_pq = scipy.linalg.solve_triangular(low_node.T, buf_pq, lower=False,
                                                   check_finite=False).T
            buf_pq += buf_pq.T
        comm.Barrier()
        
        def save_pq(buf_pq):
            with h5py.File('pq.tmp', 'w') as file_pq:
                file_pq.create_dataset('pq', shape=(naoaux, naoaux), dtype='f8')
                file_pq['pq'].write_direct(buf_pq)
        if self.shared_disk:
            if irank == 0:
                save_pq(buf_pq)
        else:
            bcast_GA(buf_pq)
            if irank_shm == 0:
                save_pq(buf_pq)
        comm.Barrier()
        free_win(win_pq)
        free_win(win_low)
        print_time(['collecting PQ', time_elapsed(t0)], log)
    def get_pq_response(self):#, win_pq, pq_node):
        win_grad, grad_node = get_shared(len(self.atom_list)*3, dtype='f8')
        atom_slice = get_slice(range(nrank), job_list=self.atom_list)[irank]
        if atom_slice is not None:
            if irank_shm == 0:
                grad = grad_node
            else:
                grad = np.zeros(len(self.atom_list)*3)
            offset_atom = self.with_df.auxmol.aoslice_by_atom()
            atm0, atm1 = atom_slice[0], atom_slice[-1]
            AUX0, AUX1 = offset_atom[atm0][-2], offset_atom[atm1][-1]
            naux_list = []
            for atm_i in atom_slice:
                aux0, aux1 = offset_atom[atm_i][2:]
                naux_list.append(aux1-aux0)
            buf_PQq = np.empty(3*max(naux_list)*self.naoaux)
            buf_PQ = np.empty(((AUX1-AUX0), self.naoaux))
            with h5py.File('pq.tmp', 'r') as file_pq:
                file_pq['pq'].read_direct(buf_PQ, source_sel=np.s_[AUX0:AUX1])
            aux_idx0 = 0
            for atm_i in atom_slice:
                s0, s1, aux0, aux1 = offset_atom[atm_i]
                naux_seg = aux1 - aux0
                aux_idx1 = aux_idx0 + naux_seg
                s_slice = [s0, s1, 0, self.with_df.auxmol.nbas]
                PQq = aux_e2(auxmol=self.with_df.auxmol, intor='int2c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, hermi=0, out=buf_PQq)
                pq_tmp = buf_PQ[aux_idx0:aux_idx1]
                rank_master = (irank//comm_shm.size)*comm_shm.size
                idx0, idx1 = atm_i*3, (atm_i+1)*3
                grad[idx0:idx1] += ddot(PQq.reshape(3,-1), pq_tmp.ravel())
                aux_idx0 = aux_idx1
            if irank_shm != 0:
                Accumulate_GA_shm(win_grad, grad_node, grad)
        comm.Barrier()
        if self.nnode > 1:
            Acc_and_get_GA(var=grad_node)
        comm_shm.Barrier()
        grad = np.copy(grad_node)
        comm_shm.Barrier()
        free_win(win_grad)
        return grad
    def collect_yi(self):
        def get_yial(self, buf_recv, mo_address):
            nao = self.nao
            nocc = self.no
            comm.Barrier()            
            win_yi = create_win(self.yi_mo, comm=comm)
            win_yi.Fence()
            occ_idx = [None]*(self.no+1)
            for idx, i in enumerate(list(self.mo_list)+[self.mo_list[-1]+1]):
                occ_idx[i] = idx
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                for rank_i, mo_i in mo_address:
                    mo0, mo1 = mo_i
                    nocc_seg = mo1 - mo0
                    idx0, idx1 = occ_idx[mo0], occ_idx[mo1]
                    dim_sup = nocc_seg*self.naoaux
                    target=[al0*dim_sup*8, nao_rank*dim_sup, MPI.DOUBLE]
                    Get_GA(win_yi, buf_recv[:nao_rank*dim_sup], target_rank=rank_i, target=target)
                    self.yi_ao[idx0:idx1] = buf_recv[:nao_rank*dim_sup].reshape(nao_rank, nocc_seg, self.naoaux).transpose(1,0,2)
            comm.Barrier()
            fence_and_free(win_yi)
        #Collect Y_i(al)
        if self.direct_int is False:
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            nocc_list = []
            mo_address = []
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i != None:
                    mo0, mo1 = mo_i[0], mo_i[-1]+1
                    nocc_list.append(len(mo_i))
                    mo_address.append([rank_i, [mo0, mo1]])
            idx_break = irank%len(mo_address)
            mo_address = mo_address[idx_break:] + mo_address[:idx_break]
            mo_slice = mo_slice[irank]
            if mo_slice is None:
                self.yi_mo = None
            else:
                self.yi_mo = contigous_trans(self.yi_mo.reshape(-1, self.nao, naoaux), order=(1, 0, 2))
            ao_slice = int_prescreen.get_slice_rank(mol, self.shell_slice, aslice=True)[0][irank]
        
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                nocc_close = len(self.mo_list)
                ncore = self.no - nocc_close
                buf_recv = np.empty(nao_rank*max(nocc_list)*self.naoaux, dtype='f8')
                self.yi_ao = np.zeros((nocc_close, nao_rank, self.naoaux))
            else:
                buf_recv = None
                self.yi_ao = None
            get_yial(self, buf_recv, mo_address)
            self.yi_mo = None
            

    #Kernel
    mol = self.mol
    auxmol = self.with_df.auxmol
    ao_loc = make_loc(self.mol._bas, 'sph')
    naoaux = self.naoaux
    log = lib.logger.Logger(self.stdout, self.verbose)
    tt = time_now()
    t0 = time_now()
    self.t_jloop = np.zeros(2)
    self.t_iloop = np.zeros(2)
    self.t_dk = np.zeros(2)
    self.t_yi = np.zeros(2)
    self.t_read = np.zeros(2)
    self.t_write = np.zeros(2)
    self.t_slice = np.zeros(2)
    self.t_remote = np.zeros(2)
    self.t_close = np.zeros(2)
    #self.opt = 1

    t0 = time_now()
    transform_yi(self)
    print_time(['Yi and dK', time_elapsed(t0)], log)

    t0 = time_now()
    grad = get_pq_response(self)
    print_time(['response of PQ', time_elapsed(t0)], log)

    t0 = time_now()
    collect_yi(self)
    print_time(['collection of Yi(P)', time_elapsed(t0)], log)

    time_list = [['j loop', self.t_jloop], ['i loop', self.t_iloop], 
                 ['dK', self.t_dk], ['Yi', self.t_yi], 
                 ['reading', self.t_read], ['writing', self.t_write]]
    if self.loc_fit:
        time_list += [['ldf slicing', self.t_slice]]
    print_time(time_list, log)
    print_time(['Yi and dK', time_elapsed(tt)], log)
    if irank == 0:
        print_mem('Yi and dK', self.pid_list, log)
    comm.Barrier()
    
    if (self.direct_int) and (irank == 0):
        if self.chkfile_qcp is None:
            shutil.rmtree(self.dir_qcp)
        if self.chkfile_ti is None:
            shutil.rmtree(self.dir_ti)
    return grad
    #sys.exit()

def mp2_dferi_GA(self):
    def get_seg(aux_loc, shell_seg, aux_seg):
        if len(aux_seg) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
            for idx, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
        else:
            shell_slice = OptPartition(nrank, shell_seg, aux_seg)[0]
            if len(shell_slice) < nrank:
                shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
                for rank_i, s_i in enumerate(shell_slice):
                    if s_i is not None:
                        shell_slice[rank_i] = s_i[0]

        shell_slice = shell_slice[irank]
        if shell_slice is not None:
            s0, s1 = shell_slice[0], shell_slice[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice = aux0, aux1
            aux_idx = [None]*(self.naoaux+1)
            for idx, num in enumerate(range(aux0, aux1+1)):
                aux_idx[num] = idx
        else:
            aux_slice, shell_slice, aux_idx = None, None, None
        return aux_slice, shell_slice, aux_idx
    def df_grad(index, buff_feri, yi_tmp, buff_yal, yli_tmp):
        def get_aoseg_by_atom(ao0, ao1, aoatoms):
            aolist = list(range(ao0, ao1))
            atmlist = aoatoms[ao0:ao1].tolist()
            aolist.append(ao1)
            atmlist.append(-2)
            idx_list = []
            atm_pre = -1
            for idx, atm in enumerate(atmlist):
                if atm != atm_pre:
                    idx_list.append(idx)
                atm_pre = atm
            aoslice = []
            for idx0, idx1 in zip(idx_list[:-1], idx_list[1:]):
                aoslice.append([atmlist[idx0], [aolist[idx0], aolist[idx1]]])
            return aoslice
        t0 = time_now()
        a0, a1, b0, b1 = index
        al0, al1, be0, be1 = [ao_loc[si] for si in index]
        nao0 = al1 - al0
        nao1 = be1 - be0
        grad = np.zeros(len(self.atom_list)*3)
        s_slice = (a0, a1, b0, b1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = time_now()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buff_feri)
        alpbe = feri_tmp.transpose(0, 2, 1).reshape(nao0*self.naoaux, -1)
        self.t_eri += time_now() - t0

        t0 = time_now()
        yli_tmp[be0:be1, ncore:] += ddot(yi_tmp, alpbe).T
        self.t_yli += time_now() - t0

        for sidx, (sa0, sa1, sb0, sb1) in enumerate([[a0, a1, b0, b1], [b0, b1, a0, a1]]):
            s_slice = (sa0, sa1, sb0, sb1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
            t0 = time_now()
            feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
            self.t_eri += time_now() - t0

            t0 = time_now()
            idx_list = [None]*(self.nao+1)
            if sidx == 0:
                for idx, aoi in enumerate(range(al0, al1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(al0, al1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,cbd->a', data_tmp, yal_tmp[:,idx0:idx1])
            else:
                for idx, aoi in enumerate(range(be0, be1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(be0, be1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[idx0:idx1])
            self.t_gra += time_now() - t0

        s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = time_now()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip2_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
        self.t_eri += time_now() - t0

        t0 = time_now()
        for atm, idx in get_aoseg_by_atom(0, self.naoaux, aux_atoms):
            p0, p1 = idx
            data_tmp = feri_tmp[:, :, :, p0:p1]
            grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[:,:,p0:p1])
        self.t_gra += time_now() - t0
        return grad
    

    def collect_grad(gradient):
        if irank_shm == 0:
            win_col = create_win(gradient, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        if irank_shm == 0 and irank != 0:
            win_col.Lock(0)
            win_col.Accumulate(gradient, target_rank=0, op=MPI.SUM)
            win_col.Unlock(0)
        win_col.Fence()
        free_win(win_col)
        return gradient

    log = lib.logger.Logger(self.stdout, self.verbose)
    log.info("\nBegin MP2 derivative feri gradient...")
    tt = time_now()

    t1 = time_now()
    if self.with_df.auxmol is None:
        self.with_df.auxmol = addons.make_auxmol(self.mol, self.with_df.auxbasis)
    t0 = time_now()
    ao_atoms = np.zeros(self.nao, dtype='i')
    for atm_i, si in enumerate(self.mol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        ao_atoms[ao0:ao1] = atm_i
    aux_atoms = np.zeros(self.naoaux, dtype='i')
    for atm_i, si in enumerate(self.with_df.auxmol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        aux_atoms[ao0:ao1] = atm_i

    t0 = time_now()
    #Get integral slices and auxilary slices
    naoaux = self.naoaux
    ao_loc = make_loc(self.mol._bas, 'sph')
    auxmol = self.with_df.auxmol
    aux_loc = make_loc(auxmol._bas, 'sph')
    shell_seg = []
    naux_seg = []
    idx0 = idx1 = 0
    while idx1 < (len(aux_loc)-1):
        idx1 = idx0 + 1
        shell_seg.append([idx0, idx1])
        naux_seg.append(aux_loc[idx1]-aux_loc[idx0])
        idx0 = idx1
    aux_slice, shell_slice, aux_idx = get_seg(aux_loc, shell_seg, naux_seg)

    win_yli, yli_node = get_shared((self.nao, self.no), dtype='f8')
    win_grad, grad_node = get_shared(self.mol.natm*3)
    #Calculate gradient
    '''self.t_eri_gen = np.zeros(2)
    self.t_eri_cal = np.zeros(2)'''
    self.t_eri = np.zeros(2)
    self.t_yli = np.zeros(2)
    self.t_trans = np.zeros(2)
    self.t_gra = np.zeros(2)
    self.t_read = np.zeros(2)
    if self.shell_slice is None:
        self.shell_slice = int_prescreen.shell_prescreen(self.mol, self.with_df.auxmol, log, self.shell_slice, 
                                                   self.shell_tot, meth_type='MP2')
    self.yal_type = 0
    ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.mol, self.shell_slice, aslice=True)
    max_memory = get_mem_spare(self.mol, 0.9)
    if shell_slice_rank is not None:
        if irank_shm == 0:
            grad_tmp = grad_node
        else:
            grad_tmp = np.zeros(len(self.atom_list)*3)
        atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
        nocc_close = len(self.mo_list)
        ncore = self.no - nocc_close
        size_yi, size_feri, shell_slice_rank = mem_control(self.mol, nocc_close, self.naoaux, shell_slice_rank, 
                                                           "derivative_feri", max_memory)
        loop_list = slice2seg(self.mol, shell_slice_rank)

        if irank_shm ==0:
            yli_tmp = yli_node
        else:
            yli_tmp = np.zeros((self.nao, self.no))
        
        '''buff_feri = np.empty((3*max(naop_list)*self.naoaux))
        buff_yal = np.empty(max(naop_list)*self.naoaux)'''
        buff_int = np.empty(size_feri)
        buff_feri = buff_int[:(size_feri*3//4)]
        buff_yal = buff_int[(size_feri*3//4):]
        if self.direct_int:
            #buff_yi = np.empty((nocc_close*max(nal_list)*self.naoaux), dtype='f8')
            buff_yi = np.empty(size_yi, dtype='f8')
        else:
            buff_yi = None
        #for int_i in int_slice:
        occ_idx = [None]*(self.no+1)
        for idx, i in enumerate(list(self.mo_list) + [self.mo_list[-1]+1]):
            occ_idx[i] = idx
        ao_idx = [None]*(self.nao+1)
        al0, al1 = ao_slice[irank]
        for idx, al in enumerate(range(al0, al1+1)):
            ao_idx[al] = idx
        mo_coeff = self.o[:, ncore:]
        for a0, a1, be_list in loop_list:
            #Get y_ialp
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0
            if self.direct_int:
                t0 = time_now()
                yi_tmp = buff_yi[:nocc_close*nao0*self.naoaux].reshape(nocc_close, nao0, self.naoaux)
                mo_slice = get_slice(range(nrank), job_list=self.mo_list)
                mo_address = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i != None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_address.append([rank_i, [mo0, mo1]])
                idx_break = irank%len(mo_address)
                mo_address = mo_address[idx_break:] + mo_address[:idx_break]

                for rank_i, mo_i in mo_address:
                    mo0, mo1 = mo_i
                    idx0, idx1 = occ_idx[mo0], occ_idx[mo1]
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, rank_i), 'r') as f:
                        #f['yi'].read_direct(yi_tmp, source_sel=np.s_[:, aux0:aux1], dest_sel=np.s_[mo0:mo1])
                        f['yi'].read_direct(yi_tmp[idx0:idx1], source_sel=np.s_[:, al0:al1])
                self.t_read += time_now() - t0
                yi_tmp = yi_tmp.reshape(nocc_close, -1)
            else:
                idx0, idx1 = ao_idx[al0], ao_idx[al1]
                yi_tmp = self.yi_ao[:,idx0:idx1].reshape(nocc_close, -1)
            for (b0, b1) in be_list:
                be0, be1 = ao_loc[b0], ao_loc[b1]
                nao1 = be1 - be0
                t0 = time_now()
                yal_tmp = buff_yal[:nao1*nao0*self.naoaux].reshape(nao1, -1)
                yal_tmp = ddot(mo_coeff[be0:be1], yi_tmp, out=yal_tmp).reshape(nao1, nao0, self.naoaux)
                self.t_trans += time_now() - t0
                int_i = (a0, a1, b0, b1)
                grad_tmp += df_grad(int_i, buff_feri, yi_tmp, yal_tmp, yli_tmp)
        if self.direct_int is False:
            self.yi_ao = None
        if irank_shm != 0:
            Accumulate_GA_shm(win_grad, grad_node, grad_tmp)
            #if self.direct_int:
            Accumulate_GA_shm(win_yli, yli_node, yli_tmp)
    buff_int = None
    buff_feri = None
    buff_yal = None
    buff_yi = None
    comm.Barrier()
    
    #if self.direct_int:
    Acc_and_get_GA(yli_node)
    if irank_shm == 0: 
        self.Yli = ddot(self.mo_coeff.T, yli_node)
    comm_shm.Barrier(); free_win(win_yli)
    if self.direct_int and irank == 0:
        shutil.rmtree(self.dir_yi)
    Gamma_omuG = None
    #Collect contribution of gradient from different nodes
    t_syn = time_now()
    if self.nnode > 1:
        Acc_and_get_GA(grad_node)
    comm_shm.Barrier()
    grad = np.copy(grad_node)
    comm_shm.Barrier()
    free_win(win_grad)
    t_syn = time_now() - t_syn

    if irank == 0:
        time_list = [['yli', self.t_yli], ['feri', self.t_eri], ['back trans', self.t_trans], ['grad', self.t_gra]]
        if self.direct_int:
            time_list.append(['reading', self.t_read])
        print_time(time_list, log)
        print_time(['MP2 derivarive feri', time_elapsed(tt)], log)
        print_mem('MP2 derivarive feri', self.pid_list, log)
    return grad

def dfhf_response_ga(self, dm1, dm2, feri):#ialp, feri):
    def get_gamma_omug(dm1, dm2):
        naoaux= self.naux_hf
        win_a, A = get_shared(naoaux, dtype='f8')
        win_b, B = get_shared(naoaux, dtype='f8')
        if self.direct_int:
            auxmol = self.with_df.auxmol
            naoaux = self.naux_hf
            ao_loc = make_loc(self.mol._bas, 'sph')
            if self.shell_slice is None:
                self.shell_slice = int_prescreen.shell_prescreen(self.mol, auxmol, log, shell_slice=self.shell_slice, 
                                                               shell_tot=self.shell_tot, meth_type='RHF')
            shellslice_rank = int_prescreen.get_slice_rank(self.mol, self.shell_slice)
            max_memory = get_mem_spare(mol, 0.9)
            if shellslice_rank is not None:
                if irank_shm != 0:
                    A_tmp = np.zeros(naoaux)
                    B_tmp = np.zeros(naoaux)
                else:
                    A_tmp = A
                    B_tmp = B
                '''naop_list = []
                for si in shellslice_rank:
                    a0, a1, b0, b1 = si
                    naop_list.append((ao_loc[a1]-ao_loc[a0])*(ao_loc[b1]-ao_loc[b0]))
                buf_feri = np.empty(max(naop_list)*naoaux)'''
                size_feri, shellslice_rank = mem_control(self.mol, self.no, naoaux, 
                                                         shellslice_rank, 0.9, max_memory)
                buf_feri = np.empty(size_feri)
                for idx, si in enumerate(shellslice_rank):
                    a0, a1, b0, b1 = si
                    al0, al1 = ao_loc[a0], ao_loc[a1]
                    be0, be1 = ao_loc[b0], ao_loc[b1]
                    nao0, nao1 = ao_loc[a1]-ao_loc[a0], ao_loc[b1]-ao_loc[b0]
                    #s_slice = (a0, a1, b0, b1, self.mol.nbas, self.mol.nbas+auxmol.nbas)
                    s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+auxmol.nbas)
                    feri_tmp = aux_e2(self.mol, auxmol, intor='int3c2e_sph', aosym='s1', comp=1, shls_slice=s_slice, out=buf_feri).transpose(1,0,2)
                    dm1_tmp = dm1[al0:al1, be0:be1].ravel()
                    dm2_tmp = dm2[al0:al1, be0:be1].ravel()
                    A_tmp += ddot(dm1_tmp, feri_tmp.reshape(-1, naoaux))
                    B_tmp += ddot(dm2_tmp, feri_tmp.reshape(-1, naoaux))
                if irank_shm != 0:
                    Accumulate_GA_shm(win_a, A, A_tmp)
                    Accumulate_GA_shm(win_b, B, B_tmp)
            comm_shm.Barrier()
        else:
            #aux_slice = get_slice(range(nrank), job_size=self.naoaux)[irank]
            aux_slice = get_auxshell_slice(self.with_df.auxmol)[0][irank]
            if aux_slice is not None:
                feri_buffer_unpack = np.empty((self.nao, self.nao))
                if (self.outcore):
                    with h5py.File(self.feri_aux, 'r') as feri_aux:
                        for idx, num in enumerate(aux_slice):
                            lib.numpy_helper.unpack_tril(np.asarray(feri_aux['j3c'][idx]), out=feri_buffer_unpack)
                            A[num] = ddot(feri_buffer_unpack.ravel(), dm1.ravel())
                            B[num] = ddot(feri_buffer_unpack.ravel(), dm2.ravel())
                else:
                    for idx, num in enumerate(aux_slice):
                        lib.numpy_helper.unpack_tril(self.feri_aux[idx], out=feri_buffer_unpack)
                        A[num] = ddot(feri_buffer_unpack.ravel(), dm1.ravel())
                        B[num] = ddot(feri_buffer_unpack.ravel(), dm2.ravel())
                feri_buffer_unpack = None
                self.feri_aux = None
        comm.Barrier()
        
        Acc_and_get_GA(A)
        Acc_and_get_GA(B)
        if irank_shm == 0:
            if self.direct_int:
                with h5py.File('j2c_hf.tmp', 'r') as f:
                    j2c = np.asarray(f['j2c'])
                    scipy.linalg.solve(j2c, A, overwrite_b=True)
                    scipy.linalg.solve(j2c, B, overwrite_b=True)
            else:
                with h5py.File('j2c_hf.tmp', 'r') as f:
                    low = np.asarray(f['low'])
                    scipy.linalg.solve_triangular(low.T, A, lower=False, overwrite_b=True, 
                                                  check_finite=False)
                    scipy.linalg.solve_triangular(low.T, B, lower=False, overwrite_b=True, 
                                                  check_finite=False)
        
        return win_a, A, win_b, B

    def get_seg(aux_loc, shell_seg, aux_seg):
        if len(aux_seg) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
            for idx, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
        else:
            shell_slice = OptPartition(nrank, shell_seg, aux_seg)[0]
            if len(shell_slice) < nrank:
                shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
                for rank_i, s_i in enumerate(shell_slice):
                    if s_i is not None:
                        shell_slice[rank_i] = s_i[0]
        shell_slice = shell_slice[irank]
        if shell_slice is not None:
            s0, s1 = shell_slice[0], shell_slice[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice = aux0, aux1
            aux_idx = [None]*(self.naoaux+1)
            for idx, num in enumerate(range(aux0, aux1+1)):
                aux_idx[num] = idx
        else:
            aux_slice, shell_slice, aux_idx = None, None, None
        return aux_slice, shell_slice, aux_idx

    def get_aoseg_by_atom(ao0, ao1, aoatoms):
        aolist = list(range(ao0, ao1))
        atmlist = aoatoms[ao0:ao1].tolist()
        aolist.append(ao1)
        atmlist.append(-2)
        idx_list = []
        atm_pre = -1
        for idx, atm in enumerate(atmlist):
            if atm != atm_pre:
                idx_list.append(idx)
            atm_pre = atm
        aoslice = []
        for idx0, idx1 in zip(idx_list[:-1], idx_list[1:]):
            aoslice.append([atmlist[idx0], [aolist[idx0], aolist[idx1]]])
        return aoslice

    def df_grad(index, buff_feri, yal_tmp):
        a0, a1, b0, b1 = index
        al0, al1, be0, be1 = [ao_loc[s] for s in index]
        nao0 = al1 - al0
        nao1 = be1 - be0
        grad = np.zeros(len(self.atom_list)*3)
        
        for sidx, (sa0, sa1, sb0, sb1) in enumerate([[a0, a1, b0, b1], [b0, b1, a0, a1]]):
            s_slice = (sa0, sa1, sb0, sb1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
            t0 = time_now()
            feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
            self.t_eri += time_now() - t0

            t0 = time_now()
            idx_list = [None]*(self.nao+1)
            if sidx == 0:
                for idx, aoi in enumerate(range(al0, al1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(al0, al1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,cbd->a', data_tmp, yal_tmp[:,idx0:idx1])
            else:
                for idx, aoi in enumerate(range(be0, be1+1)):
                    idx_list[aoi] = idx
                for atm, idx in get_aoseg_by_atom(be0, be1, ao_atoms):
                    ao0, ao1 = idx
                    idx0, idx1 = idx_list[ao0], idx_list[ao1]
                    data_tmp = feri_tmp[:, idx0:idx1]
                    grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[idx0:idx1])
            self.t_gra += time_now() - t0

        s_slice = (b0, b1, a0, a1, self.mol.nbas, self.mol.nbas+self.with_df.auxmol.nbas)
        t0 = time_now()
        feri_tmp = aux_e2(self.mol, self.with_df.auxmol, intor='int3c2e_ip2_sph', aosym='s1', comp=3, shls_slice=s_slice, out=buff_feri)
        self.t_eri += time_now() - t0

        t0 = time_now()
        for atm, idx in get_aoseg_by_atom(0, self.naoaux, aux_atoms):
            p0, p1 = idx
            data_tmp = feri_tmp[:, :, :, p0:p1]
            grad[atm_seg[atm]] -= np.einsum('abcd,bcd->a', data_tmp, yal_tmp[:,:,p0:p1])
        self.t_gra += time_now() - t0
        return grad

    def collect_grad(gradient):
        if irank_shm == 0:
            win_col = create_win(gradient, comm=comm)
        else:
            win_col = create_win(None, comm=comm)
        if irank_shm == 0 and irank != 0:
            win_col.Lock(0)
            win_col.Accumulate(gradient, target_rank=0, op=MPI.SUM)
            win_col.Unlock(0)
        win_col.Fence()
        free_win(win_col)
        return gradient

    tt = time_now()
    t1 = time_now()
    self.naoaux = self.naux_hf
    self.atom_list = range(self.mol.natm)
    if self.with_df.auxmol is None:
        self.with_df.auxmol = addons.make_auxmol(self.mol, self.with_df.auxbasis)
    log = lib.logger.Logger(self.stdout, self.verbose)
    nocc = self.no
    nao = self.nao
    naoaux = self.naux_hf
    ao_loc = make_loc(self.mol._bas, 'sph')
    mol = self.mol
    auxmol = self.with_df.auxmol
    aux_loc = make_loc(auxmol._bas, 'sph')
    ao_atoms = np.zeros(self.nao, dtype='i')
    for atm_i, si in enumerate(self.mol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        ao_atoms[ao0:ao1] = atm_i
    aux_atoms = np.zeros(self.naoaux, dtype='i')
    for atm_i, si in enumerate(self.with_df.auxmol.aoslice_by_atom()):
        ao0, ao1 = si[2:]
        aux_atoms[ao0:ao1] = atm_i
    atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
    #if self.direct_int:
    def transform_yi(self, A, B):
        mo_slice = get_slice(range(nrank), job_size=self.no)
        mo_address = [None]*self.no
        for rank_i, mo_i in enumerate(mo_slice):
            if mo_i is not None:
                for idx, i in enumerate(mo_i):
                    mo_address[i] = [rank_i, [idx, idx+1]]
        if self.direct_int:
            self.dir_yi = "yi_hf_tmp"
            self.dir_cal = '%s/cal'%self.dir_yi
            if irank == 0:
                for dir_i in [self.dir_yi, self.dir_cal]:
                    make_dir(dir_i)
            comm.Barrier()
        mo_slice = mo_slice[irank]
        win_pq, pq_node = get_shared((naoaux, naoaux))
        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            #read_file('j2c_hf.tmp', 'low_inv', buffer=low_node)
            read_file('j2c_hf.tmp', 'low', buffer=low_node)
        comm_shm.Barrier()
        if mo_slice is not None:
            if irank_shm == 0:
                buf_pq = pq_node
            else:
                buf_pq = np.zeros((naoaux, naoaux))
            nocc_rank = len(mo_slice)
            #occ_coeff = self.mo_coeff[:,:self.no]
            occ_coeff = self.o
            if self.direct_int:
                file_yi = h5py.File('%s/yi_%d.tmp'%(self.dir_yi, irank), 'w')
                #file_yi.create_dataset('yi', shape=(nocc_rank, naoaux, nao), dtype='f8')
                file_yi.create_dataset('yi', shape=(nocc_rank, nao, naoaux), dtype='f8')
                yi = np.empty((self.nao, naoaux))
                ialp_i = np.empty((self.nao, naoaux))
            else:
                self.yi_mo = np.empty((nocc_rank, self.nao, naoaux))
                
            t_feri = np.zeros(2)
            for idx_i, i in enumerate(mo_slice):
                #Read ialp_i
                if self.direct_int:
                    file_ialp = '%s/ialp_%d.tmp'%(self.dir_ialp, i)
                    read_file(file_ialp, 'ialp', buffer=ialp_i)
                    #os.remove(file_ialp)
                else:
                    ialp_i = self.ialp_mo[idx_i]
                    yi = self.yi_mo[idx_i]
                #Compute yi
                ddot(dm2, ialp_i, out=yi)
                
                #compute PQ
                buf_pq -= ddot(ialp_i.T, yi)
                scipy.linalg.solve_triangular(low_node.T, yi.T, lower=False, overwrite_b=True, 
                                              check_finite=False)
                #Save yi
                if self.direct_int:
                    file_yi['yi'].write_direct(yi, dest_sel=np.s_[idx_i])
            if irank_shm != 0:
                Accumulate_GA_shm(win_pq, pq_node, buf_pq)
                buf_pq = None
        else:
            buf_pq = None
        comm.Barrier()
        Accumulate_GA(var=pq_node)
        if irank == 0:
            scipy.linalg.solve_triangular(low_node.T, buf_pq.T, lower=False, overwrite_b=True, 
                                          check_finite=False)
            buf_pq = scipy.linalg.solve_triangular(low_node.T, buf_pq, lower=False,
                                                   check_finite=False).T
            buf_pq += ddot(A.reshape(-1, 1), B.reshape(1, -1))#np.einsum('i, j->ij', A, B)
            buf_pq += buf_pq.T
            with h5py.File('pq.tmp', 'w') as file_pq:
                file_pq.create_dataset('pq', shape=(naoaux, naoaux), dtype='f8')
                file_pq['pq'].write_direct(buf_pq)
        comm.Barrier()
        free_win(win_pq)
        free_win(win_low)


    def get_pq_response(self):#, pq_node):
        win_grad, grad_node = get_shared(len(self.atom_list)*3, dtype='f8')
        atom_slice = get_slice(range(nrank), job_list=self.atom_list)[irank]
        if atom_slice is not None:
            if irank_shm == 0:
                grad = grad_node
            else:
                grad = np.zeros(len(self.atom_list)*3)
            offset_atom = self.with_df.auxmol.aoslice_by_atom()
            atm0, atm1 = atom_slice[0], atom_slice[-1]
            AUX0, AUX1 = offset_atom[atm0][-2], offset_atom[atm1][-1]
            naux_list = []
            for atm_i in atom_slice:
                aux0, aux1 = offset_atom[atm_i][2:]
                naux_list.append(aux1-aux0)
            buf_PQq = np.empty(3*max(naux_list)*naoaux)
            buf_PQ = np.empty(((AUX1-AUX0), naoaux))
            with h5py.File('pq.tmp', 'r') as file_pq:
                file_pq['pq'].read_direct(buf_PQ, source_sel=np.s_[AUX0:AUX1])
            aux_idx0 = 0
            for atm_i in atom_slice:
                s0, s1, aux0, aux1 = offset_atom[atm_i]
                naux_seg = aux1 - aux0
                aux_idx1 = aux_idx0 + naux_seg
                s_slice = [s0, s1, 0, self.with_df.auxmol.nbas]
                PQq = aux_e2(auxmol=self.with_df.auxmol, intor='int2c2e_ip1_sph', aosym='s1', comp=3, shls_slice=s_slice, hermi=0, out=buf_PQq)
                pq_tmp = buf_PQ[aux_idx0:aux_idx1]
                idx0, idx1 = atm_i*3, (atm_i+1)*3
                grad[idx0:idx1] += ddot(PQq.reshape(3,-1), pq_tmp.ravel())
                aux_idx0 = aux_idx1
            if (irank_shm != 0):
                Accumulate_GA_shm(win_grad, grad_node, grad)
        comm.Barrier()
        return win_grad, grad_node
    #Collect Yi
    def collect_yi(self):
        def get_yial(self, buf_recv, mo_address):
            nao = self.nao
            nocc = self.no
            comm.Barrier()            
            win_yi = create_win(self.yi_mo, comm=comm)
            win_yi.Fence()
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                for rank_i, mo_i in mo_address:
                    mo0, mo1 = mo_i
                    nocc_seg = mo1 - mo0
                    dim_sup = nocc_seg*self.naoaux
                    target=[al0*dim_sup*8, nao_rank*dim_sup, MPI.DOUBLE]
                    Get_GA(win_yi, buf_recv[:nao_rank*dim_sup], target_rank=rank_i, target=target)
                    self.yi_ao[mo0:mo1] = buf_recv[:nao_rank*dim_sup].reshape(nao_rank, nocc_seg, self.naoaux).transpose(1,0,2)
            comm.Barrier()
            fence_and_free(win_yi)
        #Collect Y_i(al)
        if self.direct_int is False:
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            nocc_list = []
            mo_address = []
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i != None:
                    mo0, mo1 = mo_i[0], mo_i[-1]+1
                    nocc_list.append(len(mo_i))
                    mo_address.append([rank_i, [mo0, mo1]])
            idx_break = irank%len(mo_address)
            mo_address = mo_address[idx_break:] + mo_address[:idx_break]
            mo_slice = mo_slice[irank]
            if mo_slice is None:
                self.yi_mo = None
            else:
                self.yi_mo = contigous_trans(self.yi_mo.reshape(-1, self.nao, naoaux), order=(1, 0, 2))
            ao_slice = int_prescreen.get_slice_rank(self.mol, self.shell_slice, aslice=True)[0][irank]
        
            if ao_slice is not None:
                al0, al1 = ao_slice
                nao_rank = al1 - al0
                buf_recv = np.empty(nao_rank*max(nocc_list)*self.naoaux, dtype='f8')
                self.yi_ao = np.zeros((self.no, nao_rank, self.naoaux))
            else:
                buf_recv = None
                self.yi_ao = None
            get_yial(self, buf_recv, mo_address)
            self.yi_mo = None

    t0 = time_now()
    win_a, A, win_b, B = get_gamma_omug(dm1, dm2)
    print_time(['fitting term A and B', time_elapsed(t0)], log)

    #win_pq, pq_node = transform_yi(self, A, B)
    t0 = time_now()
    transform_yi(self, A, B)
    collect_yi(self)
    print_time(['Yi', time_elapsed(t0)], log)

    t0 = time_now()
    win_grad, grad_node = get_pq_response(self)#, pq_node)
    print_time(['gradient of (P|Q)', time_elapsed(t0)], log)

    
    shell_seg = []
    naux_seg = []
    for s0 in range(auxmol.nbas):
        s1 = s0 + 1
        shell_seg.append([s0, s1])
        naux_seg.append(aux_loc[s1]-aux_loc[s0])
    #aux_slice, shell_slice, aux_idx = get_seg(aux_loc, shell_seg, naux_seg)
    aux_slice, aux_address, shell_slice = get_auxshell_slice(auxmol)
    aux_slice = aux_slice[irank]
    if aux_slice is not None:
        aux_idx = [None]*(naoaux+1)
        for idx, p in enumerate(aux_slice + [aux_slice[-1]+1]):
            aux_idx[p] = idx
    shell_slice = shell_slice[irank]
    t_omug = time_now() - t1
    ao_slice, shell_slice_rank = int_prescreen.get_slice_rank(self.mol, self.shell_slice, aslice=True)
    max_memory = get_mem_spare(mol, 0.9)
    if shell_slice_rank is not None:
        size_yi, size_feri, shell_slice_rank = mem_control(self.mol, self.no, self.naoaux, 
                                                           shell_slice_rank, "derivative_feri", max_memory)
        loop_list = slice2seg(self.mol, shell_slice_rank)
        buff_int = np.empty(size_feri)
        buff_feri = buff_int[:(size_feri*3//4)]
        buff_yal = buff_int[(size_feri*3//4):]
        if self.direct_int:
            #buff_yi = np.empty((self.no*max(nal_list)*self.naoaux), dtype='f8')
            buff_yi = np.empty(size_yi, dtype='f8')
        else:
            buff_yi = None
        ao_idx = [None]*(self.nao+1)
        al0, al1 = ao_slice[irank]
        for idx, al in enumerate(range(al0, al1+1)):
            ao_idx[al] = idx
        #Calculate gradient
        atm_seg = np.arange(len(self.atom_list)*3).reshape(-1, 3)
        grad_tmp = np.zeros(len(self.atom_list)*3)
        mo_coeff = -2*self.o

        self.t_eri = np.zeros(2)
        self.t_trans = np.zeros(2)
        self.t_gra = np.zeros(2)
        self.t_read = np.zeros(2)

        
        for a0, a1, be_list in loop_list:
            #Get y_ialp
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0
            if self.direct_int:
                t0 = time_now()
                yi_tmp = buff_yi[:self.no*nao0*self.naoaux].reshape(self.no, nao0, self.naoaux)
                mo_slice = get_slice(range(nrank), job_list=range(self.no))
                mo_address = []
                for rank_i, mo_i in enumerate(mo_slice):
                    if mo_i != None:
                        mo0, mo1 = mo_i[0], mo_i[-1]+1
                        mo_address.append([rank_i, [mo0, mo1]])
                idx_break = irank%len(mo_address)
                mo_address = mo_address[idx_break:] + mo_address[:idx_break]

                for rank_i, mo_i in mo_address:
                    mo0, mo1 = mo_i
                    with h5py.File('%s/yi_%d.tmp'%(self.dir_yi, rank_i), 'r') as f:
                        f['yi'].read_direct(yi_tmp[mo0:mo1], source_sel=np.s_[:, al0:al1])
                self.t_read += time_now() - t0
                yi_tmp = yi_tmp.reshape(self.no, -1)
            else:
                idx0, idx1 = ao_idx[al0], ao_idx[al1]
                yi_tmp = self.yi_ao[:,idx0:idx1].reshape(self.no, -1)
            for (b0, b1) in be_list:
                be0, be1 = ao_loc[b0], ao_loc[b1]
                nao1 = be1 - be0
                t0 = time_now()
                yal_tmp = buff_yal[:nao1*nao0*self.naoaux].reshape(nao1, -1)
                yal_tmp = ddot(mo_coeff[be0:be1], yi_tmp, out=yal_tmp).reshape(-1, self.naoaux)
                yal_tmp += ddot(dm2[be0:be1, al0:al1].reshape(-1,1), A.reshape(1,-1))
                yal_tmp += ddot(dm1[be0:be1, al0:al1].reshape(-1,1), B.reshape(1,-1))
                yal_tmp = yal_tmp.reshape(nao1, nao0, self.naoaux)
                self.t_trans += time_now() - t0
                int_i = (a0, a1, b0, b1)
                grad_tmp += df_grad(int_i, buff_feri, yal_tmp)
        if self.direct_int is False:
            self.yi_ao = None
        
        Accumulate_GA_shm(win_grad, grad_node, grad_tmp)
    buff_int = None
    buff_feri = None
    buff_yal = None
    buff_yi = None
    comm.Barrier()
    
    for win in [win_a, win_b]:
        free_win(win)
    A, B, Gamma_omuG = None, None, None
    
    #Collect contribution of gradient from different nodes
    t_syn = time_now()
    if self.nnode > 1:
        #gradient = collect_grad(gradient)
        Acc_and_get_GA(var=grad_node)
    comm_shm.Barrier()
    grad = np.copy(grad_node)
    comm_shm.Barrier()
    free_win(win_grad)
    t_syn = time_now() - t_syn

    if irank == 0:
        time_list = [['feri', self.t_eri], ['back trans', self.t_trans], ['grad', self.t_gra]]
        if self.direct_int:
            time_list.append(['reading', self.t_read])
        print_time(time_list, log)
        #print_time(['RHF gradient', time_elapsed(tt)], log)
        if self.direct_int:
            for dir_i in [self.dir_yi, self.dir_ialp]:
                shutil.rmtree(dir_i)
    return grad
