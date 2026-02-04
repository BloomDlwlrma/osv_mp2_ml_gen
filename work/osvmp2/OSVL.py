import sys
import h5py
import os 
import numpy as np
from numpy.linalg import multi_dot
import scipy
import shutil
from decimal import Decimal
import itertools
if sys.version_info[0] >= 3:
    from functools import reduce
from pyscf import lib
from pyscf.df import addons, DF
from pyscf.lib import logger
from pyscf.gto.moleintor import make_loc
from osvmp2.loc.loc_addons import *#get_ncore, localization, LMO_domains, AO_domains
from osvmp2.int_prescreen import shell_prescreen
from osvmp2.ga_addons import get_shared, free_win, parallel_feri_GA, get_ialp_GA, \
                             OSV_generation_GA, update_qmat_ml, get_sf_GA, get_sratio_GA, \
                             get_precon_GA, get_precon_by_mo_GA, get_ijp_GA, get_imup_GA, \
                             get_kmatrix_GA, update_sf_ml
from osvmp2.osvutil import ddot, time_now, time_elapsed, mem_node, make_dir, \
                           get_slice, get_buff_len, aux_e2, get_auxshell_slice, \
                           print_time, print_mem, print_align, generation_SuperMat, \
                           flip_ij, OptPartition, get_mem_spare, svd, eigh
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

def parallel_eri(self, df_obj, meth_type, log):
    if df_obj.auxmol is None:
        df_obj.auxmol = addons.make_auxmol(self.mol, df_obj.auxbasis)
    auxmol = df_obj.auxmol
    naoaux = auxmol.nao_nr()
    nao = self.nao = self.mol.nao_nr()
    nao_pair = self.nao_pair = nao * (nao+1) // 2
    log.info('Begin %s 3c2e AO integrals computation...'%meth_type)
    size_feri = naoaux*self.nao*self.nao*8*1e-6
    if size_feri/self.nnode > 0.1*self.mol.max_memory:
        outcore = True
    else:
        outcore = False
    if (self.use_ga) and (outcore==False) and (self.outcore==False):
        parallel_feri_GA(self, df_obj, meth_type, log)
    else:
        def gen_feri(index, feri_save, feri_buffer, low_node):
            s0, s1, p0, p1 = index
            s_slice = (s0, s1, 0, self.mol.nbas, self.mol.nbas, self.mol.nbas+auxmol.nbas)
            feri_tmp = aux_e2(self.mol, auxmol, intor='int3c2e_sph', aosym='s2ij', comp=1, shls_slice=s_slice, out=feri_buffer).T
            #feri_tmp = scipy.linalg.solve_triangular(self.low, feri_tmp, lower=True, overwrite_b=True)
            feri_tmp = scipy.linalg.solve_triangular(low_node, feri_tmp, lower=True, overwrite_b=True, check_finite=False)
            #ddot(low_node, feri_tmp, out=feri_tmp)
            if (self.shared_disk) or (self.outcore) or (self.use_ga):
                feri_save[:, p0:p1] = feri_tmp
                #feri_save.write_direct(feri_tmp, dest_sel=np.s_[:, p0:p1])
            else:
                feri_save[:, aop_loc[s0]:aop_loc[s1]] = feri_tmp
            return feri_save

        def collect_feri():#eri_file, len_list):
            def read_slice(slice_i, blksize):
                idx0 = idx1 = 0
                block_idx = []
                block_slice = []
                while idx0 < len(slice_i):
                    idx1 = min((idx0+blksize), len(slice_i))
                    block_idx.append([idx0, idx1])
                    block_slice.append([slice_i[idx0], slice_i[idx1-1]+1])
                    idx0 = idx1
                return block_idx, block_slice
            t1=time_now()
            self.outcore = False
            if (self.outcore):
                aux_slice, self.feri_aux_address = get_auxshell_slice(auxmol)[:2]
                aux_slice = aux_slice[irank]
                if aux_slice is not None:
                    #Re-arrange order of data transmission
                    idx_break = irank%len(feri_pair_address)
                    addr_feri_aop = feri_pair_address[idx_break:] + feri_pair_address[:idx_break]
                    aux0, aux1 = aux_slice[0], aux_slice[-1]+1
                    naux_rank = len(aux_slice)
                max_aux = get_buff_len(self.mol, size_sub=self.nao_pair, ratio=0.6, max_len=naux_rank)
                if aux_slice is not None:
                    seg_idx = np.append(np.arange(aux0, aux1, step=max_aux), aux1)
                    aux_seg = [[p0, p1] for p0, p1 in zip(seg_idx[:-1], seg_idx[1:])]
                    aux_idx = [None]*(naoaux+1)
                    for idx, p in enumerate(list(aux_slice)+[aux1]):
                        aux_idx[p] = idx
                    feri_buffer = np.empty((max_aux*self.nao_pair), dtype='f8')
                    self.feri_aux = "%s/feri_tmp_%d.tmp"%(self.dir_feri, irank)
                    with h5py.File(self.feri_aux, 'w') as file_feri_aux:
                        file_feri_aux.create_dataset('j3c', shape=(len(aux_slice), self.nao_pair), dtype='f8')#, chunks=True)
                        for p0, p1 in aux_seg:
                            naux_seg = p1 - p0
                            feri_tmp = feri_buffer[:naux_seg*self.nao_pair].reshape(naux_seg, self.nao_pair)
                            for file_name, pair_i in addr_feri_aop:
                                aop0, aop1 = pair_i
                                with h5py.File(file_name, 'r') as file_feri_aop:
                                    '''file_feri_aop['j3c'].read_direct(feri_tmp, dest_sel=np.s_[:, aop0:aop1], 
                                                                     source_sel=np.s_[p0:p1])'''
                                    feri_tmp[:, aop0:aop1] = file_feri_aop['j3c'][p0:p1]
                            idx_p0, idx_p1 = aux_idx[p0], aux_idx[p1]
                            file_feri_aux['j3c'].write_direct(feri_tmp, dest_sel=np.s_[idx_p0:idx_p1])
            elif (self.use_ga):
                aux_slice, self.feri_aux_address = get_auxshell_slice(auxmol)[:2]
                aux_slice = aux_slice[irank]
                if aux_slice != None:
                    #Re-arrange order of data transmission
                    idx_break = irank%len(feri_pair_address)
                    addr_feri_aop = feri_pair_address[idx_break:] + feri_pair_address[:idx_break]
                    len_aux = len(aux_slice)
                    self.feri_aux = np.empty((len_aux, self.nao_pair), dtype='f8')
                    aux0, aux1 = aux_slice[0], aux_slice[-1]+1
                    for file_name, pair_i in addr_feri_aop:
                        aop0, aop1 = pair_i
                        with h5py.File(file_name, 'r') as feri_i:
                            '''feri_i['j3c'].read_direct(self.feri_aux, dest_sel=np.s_[:, aop0:aop1],
                                                      source_sel=np.s_[aux0:aux1])'''
                            self.feri_aux[:, aop0:aop1] = feri_i['j3c'][aux0:aux1]
                else:
                    self.feri_aux = None
                
            else:
                self.win_feri, feri_save = get_shared((naoaux, self.nao_pair))
                df_obj._cderi = feri_save

                file_slice = get_slice(job_size=len(feri_pair_address), rank_list=self.shm_ranklist)[irank_shm]
                if file_slice != None:
                    for f_idx in file_slice:
                        file_name, feri_idx = feri_pair_address[f_idx]
                        with h5py.File(file_name, 'r') as feri_i:
                            idx0, idx1 = feri_idx
                            feri_save[:, idx0: idx1] = feri_i['j3c']
            comm.Barrier()
            t_cpu, t_wall = time_elapsed(t1)
            log.info('AO int col: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t_cpu, t_wall, mem_node(self.pid_list)))
                
            if irank == 0:
                shutil.rmtree(dir_feri_com)

        t1=time_now()
        ao_loc = make_loc(self.mol._bas, 'sph')
        aop_loc = ao_loc*(ao_loc+1)//2
        
        
        dir_now = os.getcwd()
        self.dir_feri = '%s/ao_eri_%s_tmp'%(dir_now, meth_type)
        dir_feri_com = '%s/ao_eri_com_tmp'%dir_now
        def create_dir(dir_i):
            try:
                make_dir(dir_i)
            except OSError:
                pass
        if (self.shared_disk) or (self.use_ga):
            if irank == 0:
                for dir_i in [dir_feri_com, self.dir_feri]:
                    create_dir(dir_i)
            comm.Barrier()
        elif (self.outcore):
            if irank_shm == 0:
                for dir_i in [dir_feri_com, self.dir_feri]:
                    create_dir(dir_i)
            comm_shm.Barrier()

        #Get shell slices
        def get_eri_slice(ranklist, rank_i, shell_seg, naop_seg):
            n_rank = len(ranklist)
            if len(shell_seg) < n_rank:
                shell_slice = get_slice(rank_list=ranklist, job_list=shell_seg)
                len_list = []
                len_on_rank = 0
                for idx, s_i in enumerate(shell_slice):
                    if s_i != None:
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
        shell_seg = []
        naop_seg = []
        idx0 = idx1 = 0
        while idx1 < (len(aop_loc)-1):
            idx1 = idx0 + 1
            shell_seg.append([idx0, idx1])
            naop_seg.append(aop_loc[idx1]-aop_loc[idx0])
            idx0 = idx1
        
        feri_pair_address = []
        shell_slice, len_list, len_on_rank = get_eri_slice(range(nrank), irank, shell_seg, naop_seg)
        for rank_i, shell_i in enumerate(shell_slice):
            if shell_i != None: 
                file_name = '%s/feri_tmp_%d.tmp'%(dir_feri_com, rank_i)
                s0, s1 = shell_i[0], shell_i[-1]
                feri_idx = [aop_loc[s0], aop_loc[s1]]
                feri_pair_address.append([file_name, feri_idx])
        shell_slice = shell_slice[irank]

        #Create integral file/variable
        if (self.shared_disk) or (self.outcore) or (self.use_ga):
            if (shell_slice != None):
                feri_file = h5py.File('%s/feri_tmp_%d.tmp'%(dir_feri_com, irank), 'w')
                feri_save = feri_file.create_dataset('j3c', shape=(naoaux, len_on_rank), dtype='f8', chunks=True)
        else:
            self.win_feri_mp2, feri_save = get_shared((naoaux, nao_pair))

        win_low, low_node = get_shared((naoaux, naoaux))
        if irank_shm == 0:
            j2c = auxmol.intor('int2c2e', hermi=1)
            low_node[:] = scipy.linalg.cholesky(j2c, lower=True, overwrite_a=True)
            if irank == 0:
                with h5py.File('j2c_%s.tmp'%meth_type, 'w') as f:
                    f.create_dataset('low', shape=(naoaux, naoaux), dtype='f8')
                    f['low'].write_direct(low_node)
        comm_shm.Barrier()

        if (shell_slice != None):
            #Memory control within a rank
            s0, s1 = shell_slice[0], shell_slice[-1]
            AOP0, AOP1 = aop_loc[s0], aop_loc[s1]
            max_naop = (AOP1-AOP0)
            min_naop = max(naop_seg[s0:s1])
        else:
            max_naop = None
            min_naop = None
        max_naop = get_buff_len(self.mol, size_sub=naoaux, ratio=0.25, max_len=max_naop, min_len=min_naop)
        if (shell_slice != None):
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
            feri_buffer = np.empty((max(naop_list)*naoaux), dtype='f8')

            #Generation of 3c2e integrals
            for int_i in int_slice:
                feri_save = gen_feri(int_i, feri_save, feri_buffer, low_node)

            feri_buffer = None
            if (self.shared_disk) or (self.outcore) or (self.use_ga):
                feri_file.close()
        comm.Barrier()
        free_win(win_low); low_node=None
        t_cpu, t_wall = time_elapsed(t1)
        log.info('AO int cal: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t_cpu, t_wall, mem_node(self.pid_list)))
        print_time(['computing 3c2e integrals', time_elapsed(t1)], log)
        
        #if ((self.shared_disk) and (self.outcore == False)) or (self.use_ga):
        if (self.shared_disk) or (self.outcore) or (self.use_ga):
            t1=time_now()
            collect_feri()
            if irank == 0:
                print_time(['collecting integrals', time_elapsed(t1)], log)
        elif (self.outcore == False):
            df_obj._cderi = feri_save


def get_ialp(self, df_obj, meth_type, log):
    if (self.use_ga):
        get_ialp_GA(self, df_obj, meth_type, log)
    else:
        def collect_ialp():
            t1=time_now()
            if (self.outcore):
                self.ialp_mo = [None]*self.no
                if (self.shared_disk):
                    #mo_slice = get_slice(job_size=self.no, rank_list=range(nrank))
                    mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))
                    #self.dir_ialp_mo = '%s/ialp_mo_mp2_tmp'%os.getcwd()
                    if irank == 0:
                        try:
                            make_dir(self.dir_ialp_mo)
                        except OSError:
                            pass
                    comm.Barrier()
                    for rank_i, mo_i in enumerate(mo_slice):
                        if mo_i != None:
                            for i in mo_i:
                                self.ialp_mo[i] = rank_i
                    mo_slice = mo_slice[irank]
                else:
                    #mo_slice = get_slice(job_size=self.no, rank_list=self.shm_ranklist)
                    mo_slice = get_slice(job_list=self.mo_list, rank_list=self.shm_ranklist)
                    #self.dir_ialp_mo = '%s/ialp_mo_mp2_tmp'%os.getcwd()
                    if irank_shm == 0:
                        try:
                            make_dir(self.dir_ialp_mo)
                        except OSError:
                            pass
                    comm_shm.Barrier()
                    for rank_i, mo_i in enumerate(mo_slice):
                        rank_i = self.rank_list[rank_i]
                        if mo_i != None:
                            for i in mo_i:
                                self.ialp_mo[i] = rank_i
                    mo_slice = mo_slice[irank_shm]                   
                mem_avail = get_mem_spare(self.mol, ratio=0.6)
                if mo_slice is not None:
                    #ialp = h5py.File('%s/ialp_mo_mp2_%d.tmp'%(self.dir_ialp_mo, irank), 'w')
                    #with h5py.File('%s/ialp_mo_mp2_%d.tmp'%(self.dir_ialp_mo, irank), 'w') as ialp:
                    with h5py.File('%s/ialp_mo_%d.tmp'%(self.dir_ialp_mo, irank), 'w') as ialp:
                        #max_memory = self.mol.max_memory/len(self.shm_ranklist) - lib.current_memory()[0]
                        
                        len_mo = int(min(max(mem_avail//(naoaux*self.nao*8/1e6), 1), len(mo_slice)))
                        ialp_slice = []
                        mo0 = mo1 = mo_slice[0]
                        while mo0 <= mo_slice[-1]:
                            mo1 = min((mo0+len_mo), mo_slice[-1]+1)
                            ialp_slice.append([mo0, mo1])
                            mo0 = mo1
                        read_buffer = np.empty((len_mo, naoaux, self.nao), dtype='f8')
                        file_list, aux_seg = get_ialp_seg(self, range(naoaux), 'aux')
                        for ialp_i in ialp_slice:
                            mo0, mo1 = ialp_i
                            len_mo_i = mo1 - mo0
                            for f_idx, f_name in enumerate(file_list):
                                with h5py.File(file_list[f_idx], 'r') as f_aux:
                                    for p in aux_seg[f_idx]:
                                        read_buffer[:len_mo_i, p] = f_aux[str(p)][mo0:mo1]
                            for idx, i in enumerate(range(mo0, mo1)):
                                ialp[str(i)] = read_buffer[idx]
            else:
                self.win_ialp, self.ialp = get_shared((naoaux, self.no, self.nao))
                file_list, aux_seg = get_ialp_seg(self, range(naoaux), 'aux')
                file_slice = get_slice(job_size=len(file_list), rank_list=self.shm_ranklist)[irank_shm]
                if file_slice != None:
                    for f_idx in file_slice:
                        with h5py.File(file_list[f_idx], 'r') as ialp_i:
                            for p in aux_seg[f_idx]:
                                self.ialp[p] = ialp_i[str(p)]
            if (self.shared_disk):
                comm.Barrier()
                if (self.outcore==False):
                    if irank == 0:
                        shutil.rmtree(self.dir_ialp_aux)
            else:
                comm_shm.Barrier()
            t_cpu, t_wall = time_elapsed(t1)
            log.info('ialp col: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t_cpu, t_wall, mem_node(self.pid_list)))
        
        auxmol = df_obj.auxmol
        naoaux = auxmol.nao_nr()
        self.dir_ialp_aux = '%s/ialp_aux_%s_tmp'%(os.getcwd(), meth_type)
        self.dir_ialp_mo = '%s/ialp_mo_%s_tmp'%(os.getcwd(), meth_type)
        if (self.shared_disk):
            aux_slice = get_slice(job_size=naoaux, rank_list=range(nrank))
            if irank == 0:
                make_dir(self.dir_ialp_aux)
                make_dir(self.dir_ialp_mo)
            comm.Barrier()
            self.ialp_aux = [None]*naoaux
            for rank_i, aux_i in enumerate(aux_slice):
                if aux_i != None:
                    for p in aux_i:
                        self.ialp_aux[p] = rank_i
            aux_slice = aux_slice[irank]
        else:
            aux_slice = get_slice(job_size=naoaux, rank_list=self.shm_ranklist)
            if irank_shm == 0:
                make_dir(self.dir_ialp_aux)
                make_dir(self.dir_ialp_mo)
            comm_shm.Barrier()
            self.ialp_aux = [None]*naoaux
            for rank_i, aux_i in enumerate(aux_slice):
                if aux_i != None:
                    for p in aux_i:
                        self.ialp_aux[p] = rank_i
            aux_slice = aux_slice[irank_shm]
        t1=time_now()
        if (self.shared_disk) or (self.outcore):
            if aux_slice != None:
                with h5py.File('%s/ialp_aux_%d.tmp'%(self.dir_ialp_aux, irank), 'w') as ialp:
                    if (self.outcore):
                        feri_buffer = np.empty(self.nao_pair)
                        feri_buffer_unpack = np.empty((self.nao, self.nao))
                        with h5py.File('%s/feri_tmp_%d.tmp'%(self.dir_feri, irank), 'r') as feri:
                            for idx, num in enumerate(aux_slice):
                                feri_buffer[:] = feri['j3c'][idx]
                                lib.numpy_helper.unpack_tril(feri_buffer, out=feri_buffer_unpack)
                                ialp[str(num)] = ddot(feri_buffer_unpack, self.o).T
                    else:
                        feri_buffer_unpack = np.empty((self.nao, self.nao))
                        feri = df_obj._cderi
                        for p in aux_slice:
                            lib.numpy_helper.unpack_tril(feri[p], out=feri_buffer_unpack)
                            ialp[str(p)] = ddot(feri_buffer_unpack, self.o).T
            
            if (self.shared_disk):
                comm.Barrier()
            else:
                comm_shm.Barrier()  
        else:
            self.win_ialp, self.ialp = get_shared((naoaux, self.no, self.nao), dtype='f8')
            if aux_slice != None:
                feri = df_obj._cderi
                for p in aux_slice:
                    self.ialp[p] = ddot(lib.numpy_helper.unpack_tril(feri[p]), self.o).T
        t_cpu, t_wall = time_elapsed(t1)
        log.info('ialp cal: t_cpu: %.2f, t_wall: %.2f, mem: %.2f MB'%(t_cpu, t_wall, mem_node(self.pid_list)))
        print_time(['computing ialp', time_elapsed(t1)], log)
        
        if (self.shared_disk) or (self.outcore):
            t1=time_now() 
            collect_ialp()
            if irank == 0:
                print_time(['collecting ialp', time_elapsed(t1)], log)
        if (self.outcore) and (self.grad_cal==False):
            if (self.shared_disk):
                if irank == 0:
                    shutil.rmtree(self.dir_feri)
            else:
                if irank_shm == 0:
                    shutil.rmtree(self.dir_feri)
        


def OSV_generation(self, log):
    if (self.use_ga):
        OSV_generation_GA(self, log)
    else:
        def get_osv(num, ialp_i):
            #ialp_i = self.ialp[:, num, :]
            omumu = ddot(ialp_i.T, ialp_i)
            moint_ovv = multi_dot([self.v.T, omumu, self.v])
            bottom = 1.0/(-2*self.eo[num] + (self.ev + self.ev.reshape(-1, 1)))
            self.Ti[num][:] = moint_ovv * bottom
            self.Q_matrix_cp[num], self.s[num], v = svd(self.Ti[num])
            s_idx = (self.s[num] >= self.osv_tol)
            self.nosv[num] = sum(s_idx)
            self.nosv_cp[num] = self.Q_matrix_cp[num].shape[1]

        self.win_ti, self.Ti = get_shared((self.no, self.nv, self.nv), dtype='f8')#
        self.win_t_cp, self.Q_matrix_cp = get_shared((self.no, self.nv, self.nv), dtype='f8')#
        self.win_s, self.s = get_shared((self.no, self.nv), dtype='f8')#
        self.win_nosv, self.nosv = get_shared(self.no, dtype='i')#
        self.win_nosv_cp, self.nosv_cp = get_shared(self.no, dtype='i')#
        self.Q_matrix = [None]*self.no
        self.Q_ao = [None]*self.no
        
        mo_slice = get_slice(job_list=self.mo_list, rank_list=range(nrank))[irank]
        if mo_slice != None:
            if (self.outcore):
                for i in mo_slice:
                    with h5py.File('%s/ialp_mo_%d.tmp'%(self.dir_ialp_mo, irank), 'r') as ialp:
                        get_osv(i, np.asarray(ialp[str(i)]))
            else:
                for i in mo_slice:
                    get_osv(i, self.ialp[:,i])
        
        comm_shm.Barrier()
        dim_tco = [0]*self.no
        for i in self.mo_list:
            dim_tco[i] = self.nao*self.nosv[i]
        dim_sum = sum(dim_tco)
        self.win_t_co, T_co = get_shared(dim_sum, dtype='f8')
        loc_tco = 0
        for i in self.mo_list:
            self.Q_matrix[i] = self.Q_matrix_cp[i][:, :self.nosv[i]]
            dim_i = dim_tco[i]
            self.Q_ao[i] = T_co[loc_tco:loc_tco+dim_i].reshape(self.nao, self.nosv[i])
            loc_tco += dim_i
        mo_slice = get_slice(job_list=self.mo_list, rank_list=self.shm_ranklist)[irank_shm]
        #mo_slice = get_slice(job_size=self.no, rank_list=self.shm_ranklist)[irank_shm]
        if mo_slice is not None:
            for i in mo_slice:
                self.Q_ao[i][:] = ddot(self.v, self.Q_matrix[i])
        #if (self.outcore): eri.close()



def get_precon_by_mo(self, i):
    eigval, eigvec = eigh(self.F_matrix[i*self.no+i])
    effective_c = eigvec
    emu_i = eigval
    return effective_c, emu_i

def get_precon(self, ipair):
    i = ipair//self.no
    j = ipair%self.no
    S_mat = generation_SuperMat([i, j, i, j], self.S_matrix, self.nosv, self.no)
    eigval, eigvec = eigh(S_mat)
    S_mat = None
    newvec = eigvec[:, eigval>1e-5]/np.sqrt(eigval[eigval>1e-5])
    F_mat = generation_SuperMat([i, j, i, j], self.F_matrix, self.nosv, self.no)
    newh = multi_dot([newvec.T, F_mat, newvec])
    F_mat = None
    eigval, eigvec = eigh(newh)
    eij = self.eo[i]+self.eo[j] 
    eab = eigval+eigval.reshape(-1, 1)
    effective_c = ddot(newvec, eigvec)
    effective_d = 1.0/(eij - eab)
    return effective_c, effective_d

def get_imup(self, num, ialp_cache):
    return np.dot(ialp_cache[num], self.Q_ao[num])

def get_kmatrix(self, ipair, ialp_cache):#ialp_cache, ipair):
    def save_file(f_ij, f_ji, ipair):
        self.ijmul_save[str(ipair)] = f_ij
        self.jinul_save[str(ipair)] = f_ji
    def get_vir_coeff(self, i, j):
        return np.concatenate((self.Q_ao[i], self.Q_ao[j]), axis=1)
    i = ipair //self.no
    j = ipair % self.no
    if (self.if_remote[ipair]):
        k_mat = ddot(self.imup[i].T, self.imup[j])
    else:
        vir_coeff = get_vir_coeff(self, i, j)
        ijmuL = ddot(ialp_cache[i], vir_coeff)
        jinuL = ddot(ialp_cache[j], vir_coeff)
        k_mat = ddot(ijmuL.T, jinuL)
    if (self.use_mbe == False) and (self.grad_cal) and (self.if_remote[ipair]==False):
        if (self.outcore):
            save_file(ijmuL, jinuL, ipair)
        else:
            self.ijmuL[ipair] = ijmuL
            self.jinuL[ipair] = jinuL
    if (self.use_mbe) and (self.grad_cal) and (self.if_remote[ipair]==False):
        return ijmuL, jinuL, k_mat
    else:
        return k_mat


  
def presidue(self, pairlist, K_matrix, T_matrix):
    def check_redundant(self, i, j, k):
        def get_pair(i, j):
            if i < j:
                return i*self.no+j
            else:
                return j*self.no+i
        if (i == k) or (j == k):
            return False
        else:
            if_red = False
            pairs = [get_pair(i, k), get_pair(j, k)]
            for ipair in pairs:
                if self.if_remote[ipair] or self.if_discarded[ipair]:
                    if_red = True
                    break
            return if_red
    def residue_eval(ipair, R_matrix):
        i = ipair//self.no
        j = ipair%self.no
        R_matrix[ipair] = np.copy(K_matrix[ipair])
        if self.if_remote[ipair]:
            T_ii = T_matrix[i*self.no+i][:self.nosv[i], self.nosv[i]:]
            T_jj = T_matrix[j*self.no+j][:self.nosv[j], self.nosv[j]:]
            R_matrix[ipair] += ddot(T_matrix[ipair], (self.F_matrix[j*self.no+j] - self.loc_fock[j,j]))
            R_matrix[ipair] += ddot((self.F_matrix[i*self.no+i] - self.loc_fock[i,i]), T_matrix[ipair])
            R_matrix[ipair] -= self.loc_fock[i,j] * (ddot(self.S_matrix[i*self.no+j], T_jj) + 
                                                     ddot(T_ii, self.S_matrix[i*self.no+j]))
        else:
            k_tol = 1e-5
            F_ijij = generation_SuperMat([i, j, i, j], self.F_matrix, self.nosv, self.no)
            for k in self.mo_list:#range(self.no):
                if check_redundant(self, i, j, k): continue
                if abs(self.loc_fock[k, j]) > k_tol:
                    S_ikij = generation_SuperMat([i, k, i, j], self.S_matrix, self.nosv, self.no)
                    B = -self.loc_fock[k, j] * S_ikij
                    if (k==j):
                        B += F_ijij
                    if i > k:
                        T_ik = flip_ij(k, i, T_matrix[k*self.no+i], self.nosv)
                    else:
                        T_ik = T_matrix[i*self.no+k]
                    R_matrix[ipair] += multi_dot([S_ikij.T, T_ik, B])
                if abs(self.loc_fock[i, k]) > k_tol:
                    S_ijkj = generation_SuperMat([i, j, k, j], self.S_matrix, self.nosv, self.no)
                    C = -self.loc_fock[i, k] * S_ijkj
                    if (i==k):
                        C += F_ijij
                    if k > j:
                        T_kj = flip_ij(j, k, T_matrix[j*self.no+k], self.nosv)
                    else:
                        T_kj = T_matrix[k*self.no+j]
                    R_matrix[ipair] += multi_dot([C, T_kj, S_ijkj.T])
        return R_matrix
    R_matrix = [None]*self.no**2
    for ipair in pairlist:
        R_matrix = residue_eval(ipair, R_matrix)
    
    return R_matrix

class OSVLMP2():
    def __init__ (self, RHF, my_para):
        self.__dict__.update(my_para.__dict__)
        self.chkfile_ialp = my_para.chkfile_ialp_mp2
        self.RHF = RHF
        self.t_hf = RHF.t_hf
        self.mol = RHF.mol
        self.mo_energy = RHF.mo_energy
        self.mo_occ = RHF.mo_occ
        self.mo_coeff = RHF.mo_coeff
        self.direct_int = RHF.direct_int
        self.shell_tol = RHF.shell_tol
        self.with_solvent = self.RHF.with_solvent
        #self.solvent = my_para.solvent
        self.atom_list = range(self.mol.natm)
        self.stdout = sys.stdout
        self.naoaux = self.naux_mp2# = my_para.naux_mp2
        self.shm_ranklist = range(len(self.rank_list))
        self.nao = self.mol.nao_nr()
        self.nao_pair = self.nao * (self.nao+1) // 2
        log = lib.logger.Logger(self.stdout, self.verbose)
        self.cposv_tol = min(self.cposv_tol, self.osv_tol)
        if self.grad_cal:
            self.idsvd_tol = -1
        self.o = self.mo_coeff[:, self.mo_occ>0]
        self.v = self.mo_coeff[:, self.mo_occ==0]
        self.ev = self.mo_energy[self.mo_occ==0]
        self.ev_di = np.diag(self.ev)
        self.nv = self.v.shape[1]
        self.nao = self.v.shape[0]
        self.no = self.nao - self.nv
        if self.use_frozen == True:
          self.use_sl = True
        if self.use_sl == True:
          #self.nocc_core = get_ncore(self.mol)
          #self.nocc_core = get_ncore(self)
          self.nocc_core = get_ncore(self.mol)
        else:
          self.nocc_core = None  
        if self.osv_tol == 0:
            self.use_cposv = False
        self.lg_dr = False
        if self.grad_cal:
            self.ene_tol = 1e-8
        else:
            self.ene_tol = 1e-6
        self.ijmuL = [None]*self.no**2
        self.jinuL = [None]*self.no**2
        self.clus_type = 0
        
    def kernel(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        #log.info('\n--------------------------------MP2 energy---------------------------------')
        

        if self.ml_test and (self.ml_mp2int == False):
            self.__dict__.update(self.RHF.__dict__)
            self.t_loc = self.t_feri_mp2 = np.zeros(2)
        else:
            log.info('\n--------------------------------Localization---------------------------------')
            t1=time_now()
            #Initialise occupied mo list
            if self.use_frozen == True:
                log.info("Frozen core: ON")
                self.mo_list = range(self.no)[self.nocc_core:]
            else:
                self.mo_list = range(self.no)
                log.info("Frozen core: OFF")
            self.nocc = len(self.mo_list)
                
            #pop_method = 'mul_melow'
            #pop_method = 'low_melow'
            #pop_method = 'mulliken'
            #pop_method = 'lowdin'
            
            self.win_o, self.o = get_shared((self.nao, self.no))
            self.win_eo, self.eo = get_shared(self.no)
            self.win_loc, self.loc_fock = get_shared((self.no, self.no))
            self.win_uo, self.uo = get_shared((self.no, self.no))

            if irank_shm == 0:
                if self.chkfile_loc is not None:
                    log.info(f"Read loc matrices from check file: {self.chkfile_loc}")
                    with h5py.File(self.chkfile_loc, 'r') as f:
                        f['o'].read_direct(self.o)
                        f['eo'].read_direct(self.eo)
                        f['loc_fock'].read_direct(self.loc_fock)
                        f['uo'].read_direct(self.uo)
                else:
                    if irank == 0:
                        frozen = self.use_frozen
                        self.use_frozen = False
                    
                    if self.local_type == 0:
                        #Canonical orbital
                        self.uo[:] = np.diag(np.ones(self.no))
                        self.o[:] = self.mo_coeff[:, self.mo_occ>0]
                        self.loc_fock[:] = np.diag(self.mo_energy[:self.no])
                        self.eo[:] = self.mo_energy[:self.no]
                    else:
                        
                        self.uo[:] = localization(self.mol, self.mo_coeff[:, self.mo_occ>0], local_type=self.local_type, 
                                                    pop_method=self.pop_method, grad_cal=self.grad_cal, use_sl=self.use_sl, 
                                                    frozen=self.use_frozen, loc_fit=self.loc_fit, verbose=self.verbose, log=log)
                        ddot(self.mo_coeff[:, :self.no], self.uo, out=self.o)
                        self.loc_fock[:] = multi_dot([self.uo.T, np.diag(self.mo_energy[:self.no]), self.uo])
                        self.eo[:] = np.diag(self.loc_fock)
                        if irank == 0:
                            with h5py.File('loc_var.chk', 'w') as f:
                                f.create_dataset("uo", data=self.uo)
                                f.create_dataset("o", data=self.o)
                                f.create_dataset("loc_fock", data=self.loc_fock)
                                f.create_dataset("eo", data=self.eo)
                            if self.chkfile_save is not None:
                                if self.local_type == 1:
                                    dir_loc = "%s/pm"%self.chkfile_save
                                elif self.local_type == 2:
                                    dir_loc = "%s/boys"%self.chkfile_save
                                os.makedirs(dir_loc, exist_ok=True)
                                shutil.copy('loc_var.chk', dir_loc)
            comm_shm.Barrier()
            #if irank == 0: 
            print_time(['localization', time_elapsed(t1)], log)
            self.t_loc = time_elapsed(t1)

            t0 = time_now()
            #Generation of ialp
            log.info("\n----------------------MP2 MO integral transformation------------------------")
            #Construct ri-mp2 3c2e integrals
            self.with_df = DF(self.mol)
            self.with_df.auxbasis = self.auxbasis_mp2
            self.with_df.auxmol = self.auxmol_mp2
            self.aux_atm_offset = self.with_df.auxmol.offset_nr_by_atom()
            self.ao_atm_offset = self.mol.offset_nr_by_atom()
            if self.direct_int == False:
                parallel_eri(self, self.with_df, 'mp2', log)
                print_time(['mp2 3c2e integrals', time_elapsed(t0)], log)
                self.t_feri_mp2 = time_elapsed(t0)
            else:
                self.t_feri_mp2 = np.zeros(2)
            if (self.grad_cal) and (self.RHF.shell_slice is None):
                self.RHF.shell_slice = int_prescreen.shell_prescreen(self.mol, self.RHF.with_df.auxmol, log, 
                                    shell_slice=self.RHF.shell_slice, shell_tol=self.shell_tol, meth_type='RHF')
            self.shell_slice = int_prescreen.shell_prescreen(self.mol, self.with_df.auxmol, log, shell_slice=None, 
                                                        shell_tol=self.shell_tol, meth_type='MP2')
            t1=time_now() 
            log.info('\nBegin calculation of (ial|P)...')
            get_ialp(self, self.with_df, 'mp2', log)
            if irank == 0:
                print_mem('ialp generation', self.pid_list, log)
            print_time(['ialp generation', time_elapsed(t1)], log)
            self.t_feri_mp2 += time_elapsed(t1)
        
            if self.loc_fit:
                ave_nfit = np.sum(self.nfit)/len(self.mo_list)
                ave_nbfit = np.sum(self.nbfit)/len(self.mo_list)
                log.info('\nAverage local fitting basis for MP2 (full %d):'%self.naoaux)
                msg_list = [['Fitting (%.1E):'%self.fit_tol, int(ave_nfit)],
                            ['Block fitting (%.1E):'%self.fit_tol, int(ave_nbfit)]]
                print_align(msg_list, align='lr', indent=4, log=log)
            

        #initialize pairlists
        self.pairlist = []
        self.pairlist_redundant = []
        self.pairidx = [None]*self.no**2
        idx = 0
        for i, j in itertools.product(self.mo_list, self.mo_list):
            ipair = i*self.no+j
            if i <= j:
                self.pairlist.append(ipair)
                self.pairidx[ipair] = idx
                idx += 1
            else:
                self.pairlist_redundant.append(ipair)
        
        if irank == 0: 
            log.info("\n------------------------------OSV-based quantities-------------------------------")
        tt=time_now()
        #OSV generation
        log.info("Begin OSV generation...")
        t1=time_now()
        OSV_generation(self, log)
        
        print_time(['OSV generation', time_elapsed(t1)], log)
        if irank == 0:
            print_mem('OSV generation', self.pid_list, log)
        msg_list = [['CPOSV threshold', '%.1E'%self.cposv_tol], 
                    ['OSV threshold', '%.1E'%self.osv_tol], 
                    ['Average number of full OSVs', int(sum(self.nosv_cp)/self.nocc)],
                    ['Average number of OSVs', int(sum(self.nosv)/self.nocc)]]
        print_align(msg_list, align='lr', indent=4, log=log)
        if (sum(self.nosv)/self.nocc) == self.nv:
            self.use_cposv = False

        #Calculate S and F matrix
        t1=time_now()
        log.info("\nBegin computing S and F matrices...")
        get_sf_GA(self)
        print_time(['S/F generation', time_elapsed(t1)], log)
        if irank == 0:
            print_mem('S/F generation', self.pid_list, log)


        #Initialise pair screening (always keep diagonal pairs)        
        # Calculate overlap matrix ratio
        t0=time_now()
        self.pairlist_close = self.pairlist
        self.mo_close = self.mo_list
        self.if_remote = [False]*self.no**2
        self.if_discarded = [False]*self.no**2
        self.pairlist_remote = []
        self.pairlist_dicarded = []
        self.mo_remote = []
        
        if self.remo_tol > 0 or self.use_mbe:
            def get_sratio():
                if (self.use_ga):
                    get_sratio_GA(self)
                else:
                    self.win_s_r, self.s_ratio = get_shared(self.no**2, dtype='f8')
                    def cal_sratio(ipair):
                        i = ipair//self.no
                        j = ipair%self.no
                        pair_ji = j*self.no+i
                        self.s_ratio[pair_ji] = self.s_ratio[ipair] = sum((self.S_matrix[ipair]**2).ravel()/((self.nosv[i]+self.nosv[j])*0.5))
                    for ipair in self.pair_slice:
                        cal_sratio(ipair)
                    comm_shm.Barrier()
                for i in range(self.no):
                    self.s_ratio[i*self.no+i] = 1
            get_sratio()

            if self.remo_tol > 0:
                log.info("\nBegin pair screening...")
                self.pairlist_close = []
                self.pairlist_adjusted = []
                self.mo_close = []
                n_pair = len(self.pairlist)
                for ipair in self.pairlist:
                    i = ipair // self.no
                    j = ipair % self.no
                    if self.s_ratio[ipair] < self.disc_tol:
                        self.if_discarded[ipair] = True
                        self.if_discarded[j*self.no+i] = True
                        self.pairlist_dicarded.append(ipair)
                    else:
                        self.pairlist_adjusted.append(ipair)
                        if (self.s_ratio[ipair] < self.remo_tol):
                            self.pairlist_remote.append(ipair)
                            self.if_remote[ipair] = True
                            self.if_remote[j*self.no+i] = True
                            self.mo_remote.extend([i, j])
                        else:
                            self.pairlist_close.append(ipair)
                            self.mo_close.extend([i, j])
                self.pairlist = self.pairlist_adjusted
                self.mo_remote = sorted(set(self.mo_remote))
                self.mo_close = sorted(set(self.mo_close))
                if irank == 0: 
                    msg_list = [['Pair screening threshold', '%.1E'%self.remo_tol],
                                ['Pair discarding threshold', '%.1E'%self.disc_tol],
                                ['Number of close pairs', len(self.pairlist_close)],
                                ['Number of remote pairs', len(self.pairlist_remote)],
                                ['Number of discared pairs', len(self.pairlist_dicarded)]]
                    print_align(msg_list, align='lr', indent=4, log=log)

        self.pairlist_full = []
        self.pairlist_offdiag = []
        if self.lg_dr:
            pairlist = self.pairlist
        else:
            pairlist = self.pairlist_close
        for ipair in pairlist:
            i = ipair // self.no
            j = ipair % self.no
            self.pairlist_full.append(ipair)
            if i != j:
                self.pairlist_full.append(j*self.no+i)
                self.pairlist_offdiag.append(ipair)
        self.pairlist_full = sorted(list(set(self.pairlist_full)))
        self.refer_pairlist = self.pairlist

        # if irank_shm == 0:
        #     for ipair in self.pairlist_offdiag:
        #         i = ipair // self.no
        #         j = ipair % self.no
        #         print(i, j)
        # sys.exit()
                

        if (self.ml_test) and (self.ml_mp2int is False) and (self.nosv_ml is not None):
            msg = "\nThe number of OSVs will be fixed as %d\n"%self.nosv_ml
            msg += "    Adjusting the size of qmat and SF matrices"
            log.info(msg)
            t0=time_now()
            update_qmat_ml(self)
            update_sf_ml(self)
            print_time(['adjusting osv size', time_elapsed(t0)], log)

        #Compute K matrix
        log.info("\nStart the K matrix computations...")
        t0=time_now()
        if self.ml_test:
            get_ijp_GA(self)
        if self.mo_remote != []:
            get_imup_GA(self)
        get_kmatrix_GA(self)
        
        print_time(['K generation', time_elapsed(t0)], log)
        if irank == 0:
            print_mem('K generation', self.pid_list, log)

        if (self.ml_test is False) or (self.ml_mp2int):
            #Preconditioning
            log.info("\nStart the preconditioning...")
            t0=time_now()
            get_precon_GA(self, self.pairlist_close)
            if self.mo_remote != []:
                get_precon_by_mo_GA(self)
            print_time(['preconditioning', time_elapsed(t0)], log)
            if irank == 0:
                print_mem('preconditioning', self.pid_list, log)

        self.t_osv_gen = time_elapsed(tt)

def get_ene(self):
        t_mp2_e = time_now()
        tt=time_now()
        t_ga = np.zeros(2)
        tga = time_now()
        log = lib.logger.Logger(self.stdout, self.verbose)
        
        
    #MP2 iterations    
        if irank == 0:
            log.info("\n---------------------------MP2 residual calculation---------------------------")
        def ene_iteration(pairlist, pair_type, T_matrix, ene_list, ene_tol):#, buf_T=None, buf_R=None):
            ite = 0
            ene_old = 0.0
            ene_new = 0.0
            pairene_discard = 0.0
            converge = False
            log.info("Residue iteration for %s pairs"%pair_type)
            use_dynt = False
            if (len(pairlist) > 6) and (pair_type == "close"):
                adiis = lib.diis.DIIS()
                use_diis = True
            else:
                use_diis = False
                if pair_type is "close":
                    use_dynt = True
            #use_dynt = False
            while (not converge):
                #Dynamical amplitudes update
                if use_dynt:
                    for ipair in pairlist:
                        R_matrix = presidue(self, [ipair], self.K_matrix, T_matrix)
                        effective_R = self.emu_ij[ipair] * multi_dot([self.X_matrix[ipair].T, R_matrix[ipair], self.X_matrix[ipair]])
                        delta = multi_dot([self.X_matrix[ipair], effective_R, self.X_matrix[ipair].T])
                        T_matrix[ipair] += delta
                else:
                    R_matrix = presidue(self, pairlist, self.K_matrix, T_matrix)
                    if pair_type == "remote":
                        for ipair in pairlist:
                            i = ipair//self.no
                            j = ipair%self.no
                            eff_del = 1/(self.eo[i]+self.eo[j]-self.emu_i[i].reshape(-1,1)-self.emu_i[j].ravel())
                            effective_R = eff_del * multi_dot([self.x_ii[i].T, R_matrix[ipair], self.x_ii[j]])
                            delta = multi_dot([self.x_ii[i], effective_R, self.x_ii[j].T])
                            T_matrix[ipair] += delta
                    else:
                        for ipair in pairlist:                            
                            effective_R = self.emu_ij[ipair] * multi_dot([self.X_matrix[ipair].T, R_matrix[ipair], self.X_matrix[ipair]])
                            delta = multi_dot([self.X_matrix[ipair], effective_R, self.X_matrix[ipair].T])
                            T_matrix[ipair] += delta
                        #DIIS
                        if (use_diis):
                            if ite > 0:
                                def diis(pairlist, T_matrix, adiis):
                                    shape_ti = [None]*self.no**2
                                    size_ti = 0
                                    for ipair in pairlist:
                                        shape_ti[ipair] = T_matrix[ipair].shape
                                        size_ti += np.prod(shape_ti[ipair])
                                    T_flat = np.empty(size_ti)
                                    R_flat = np.empty(size_ti)
                                    idx1 = 0
                                    for ipair in pairlist:
                                        idx0, idx1 = idx1, idx1 + np.prod(shape_ti[ipair])
                                        T_flat[idx0:idx1] = T_matrix[ipair].ravel()
                                        R_flat[idx0:idx1] = R_matrix[ipair].ravel()

                                    T_flat = adiis.update(T_flat, R_flat)
                                    idx1 = 0
                                    for ipair in pairlist:
                                        idx0, idx1 = idx1, idx1 + np.prod(shape_ti[ipair])
                                        T_matrix[ipair] = T_flat[idx0:idx1].reshape(shape_ti[ipair])
                                    return T_matrix
                                if ite == 1: log.info("Turn on DIIS")
                                if self.use_mbe:
                                    self.buffer_tmat[:self.size_k] = adiis.update(self.buffer_tmat[:self.size_k], buf_R)
                                else:
                                    T_matrix = diis(pairlist, T_matrix, adiis)
                                #buf_T[:self.idx_close] = adiis.update(buf_T[:self.idx_close], buf_R[:self.idx_close])
                    
        #compute energies
                ene_new = 0.0
                if self.ml_test and self.mbe_mode != 2 and self.clus_type != 21:
                    ene_decom_mbe = np.zeros((len(pairlist), 9))
                for pidx, ipair in enumerate(pairlist):
                    if (self.if_remote[ipair]):
                        #T_bar_i = 2*T_matrix[ipair].ravel()
                        ene_i = 2*ddot(self.K_matrix[ipair].ravel(), T_matrix[ipair].ravel())
                        #ene_i -= ddot(self.K_matrix[ipair].T.ravel(), T_matrix[ipair].T.ravel()) 
                        if self.ml_test and self.mbe_mode != 2:
                            self.ene_decom_mbe_rank[self.idx_pair[ipair], 5] = 2 * ene_i
                    else:
                        T_bar_i = (2*T_matrix[ipair].ravel() - T_matrix[ipair].T.ravel())
                        ene_i = ddot(self.K_matrix[ipair].ravel(), T_bar_i)
                        if self.ml_test and self.mbe_mode != 2 and self.clus_type != 21:
                            #Energy decomposition
                            def split_block(i, j, mat, pos='tr'):
                                '''
                                pos = tl, tr, bl, br
                                '''
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
                                '''if self.clus_type == 1:
                                    self.ene_decom_mbe_rank[self.idx_pair[ipair], 0] = ene_i
                                    self.ene_decom_mbe_rank[self.idx_pair[ipair], 1] -= ene_i * self.count_2b[ipair]
                                    self.ene_decom_mbe_rank[self.idx_pair[ipair], 2] -= ene_i * self.count_3b[ipair]
                                elif self.clus_type == 20:
                                    self.ene_decom_mbe_rank[self.idx_pair[ipair], 1] += ene_i
                                elif self.clus_type == 3:
                                    self.ene_decom_mbe_rank[self.idx_pair[ipair], 2] += ene_i'''
                                if self.clus_type == 1:
                                    ene_decom_mbe[pidx, 0] = ene_i
                                    ene_decom_mbe[pidx, 1] -= ene_i * self.count_2b[ipair]
                                    ene_decom_mbe[pidx, 2] -= ene_i * self.count_3b[ipair]
                                elif self.clus_type == 20:
                                    ene_decom_mbe[pidx, 1] += ene_i
                                    ene_decom_mbe[pidx, 2] -= ene_i * self.count_3b[pairlist[1]]
                                elif self.clus_type == 3:
                                    ene_decom_mbe[pidx, 2] += ene_i
                            else:
                                tii = split_block(i, j, T_matrix[ipair], pos='tl')
                                tjj = split_block(i, j, T_matrix[ipair], pos='br')
                                tij = split_block(i, j, T_matrix[ipair], pos='tr')
                                tji = split_block(i, j, T_matrix[ipair], pos='bl')
                                eii = ddot(split_block(i, j, self.K_matrix[ipair], 'tl').ravel(), (2*tii-tii.T).ravel())
                                ejj = ddot(split_block(i, j, self.K_matrix[ipair], 'br').ravel(), (2*tjj-tjj.T).ravel())
                                kij = split_block(i, j, self.K_matrix[ipair], 'tr')
                                kji = split_block(i, j, self.K_matrix[ipair], 'bl')
                                eij = 2 * ddot(kij.ravel(), tij.ravel()) - ddot(kji.ravel(), tij.T.ravel())
                                eji = 2 * ddot(kji.ravel(), tji.ravel()) - ddot(kij.ravel(), tji.T.ravel())
                                #eji = ddot(split_block(i, j, self.K_matrix[ipair], 'bl').ravel(), (2*tji-tij.T).ravel())
                                tii = tjj = tij = tji = kij = kji = None
                                '''self.ene_decom_nonmbe[pidx, 1] = 2 * (eii + ejj)
                                self.ene_decom_nonmbe[pidx, 2] = 2 * eij
                                self.ene_decom_nonmbe[pidx, 3] = 2 * eji'''
                                if self.clus_type == 20:
                                    ene_decom_mbe[pidx, 3] = 2 * (eii + ejj)
                                    ene_decom_mbe[pidx, 5] = 2 * eij
                                    ene_decom_mbe[pidx, 7] = 2 * eji
                                    ene_decom_mbe[pidx, 4] -= 2 * (eii + ejj) * self.count_3b[ipair] 
                                    ene_decom_mbe[pidx, 6] -= 2 * eij * self.count_3b[ipair] 
                                    ene_decom_mbe[pidx, 8] -= 2 * eji * self.count_3b[ipair] 
                                elif self.clus_type == 3:
                                    ene_decom_mbe[pidx, 4] += 2 * (eii + ejj)
                                    ene_decom_mbe[pidx, 6] += 2 * eij
                                    ene_decom_mbe[pidx, 8] += 2 * eji
                    if (ipair//self.no) != (ipair%self.no):
                        ene_i *= 2
                    ene_new += ene_i
                    ene_list[ipair] = ene_i
                var = abs(ene_old - ene_new)
                ene_old = ene_new
                log.info('Iter. %d: energy %.10f, by energy increment %.2E', ite, ene_new, Decimal(var))   
        #converged or not
                if (var < ene_tol) or (pair_type == "remote") or (self.local_type==0):
                    converge = True
                else:
                    ite += 1            
                if(ite > self.max_cycle): 
                    log.warn('OSV-MP2 exceeds the maximum iteration number %d and will quit!', self.max_cycle)
                    converge = True
                if (converge):
                    if self.use_mbe:
                        '''if self.clus_type != 21 and self.local_type != 0:
                            #Adding back <R T_bar>
                            for ipair in pairlist:
                                T_bar_i = (2*T_matrix[ipair].ravel() - T_matrix[ipair].T.ravel())
                                ene_i = ddot(R_matrix[ipair].ravel(), T_bar_i)
                                if (ipair//self.no) != (ipair%self.no):
                                    ene_i *= 2
                                ene_list[ipair] += ene_i
                                ene_new += ene_i'''
                        self.nite_list.append(ite+1)
                    if self.ml_test and self.mbe_mode != 2 and self.clus_type != 21:
                        for pidx, ipair in enumerate(pairlist):
                            self.ene_decom_mbe_rank[self.idx_pair[ipair]] += ene_decom_mbe[pidx]
                    return ene_list, ene_new, T_matrix
        self.t_block = np.zeros(2)
        self.t_res = np.zeros(2)
        t0=time_now()
        log.info("\nStart the residue iterations...")
        converge_close, converge_remote = False, False
        ene_close, ene_remote = 0.0, 0.0

        self.size_k = 0
        for ipair in self.pairlist:
            self.size_k += self.K_matrix[ipair].size
        '''buf_R = np.zeros(self.size_k)
        R_matrix = [None]*self.no**2'''

        use_tinit = False
        if self.use_mbe:
            if (self.clus_type != 1):
                use_tinit = True
        if use_tinit:
            T_matrix = self.T_matrix
            if (self.clus_type==20) or (self.clus_type==21):
                ij = self.pairlist[1]
                T_matrix[ij] = np.zeros_like(self.K_matrix[ij])
        else:
            #buf_T = np.zeros(self.size_k)
            T_matrix = [None]*self.no**2
        
        #for idx, ipair in enumerate(self.pairlist):
        buf_idx0 = 0
        for pidx, pairlist in enumerate([self.pairlist_close, self.pairlist_remote]):
            for ipair in pairlist:
                size_k = self.K_matrix[ipair].size
                shape_k = self.K_matrix[ipair].shape
                buf_idx1 = buf_idx0 + size_k
                if use_tinit is False:
                    if self.if_remote[ipair]:
                        T_matrix[ipair] = np.zeros(shape_k)
                    else:
                        effective_R = self.emu_ij[ipair] * multi_dot([self.X_matrix[ipair].T, self.K_matrix[ipair], self.X_matrix[ipair]])
                        T_matrix[ipair] = multi_dot([self.X_matrix[ipair], effective_R, self.X_matrix[ipair].T])
                buf_idx0 = buf_idx1
            if pidx == 0:
                self.idx_close = buf_idx0
        t_ga = time_now() - tga
        t1 = time_now() 
        twob_remote = False
        if (self.use_mbe):
            if self.clus_type == 21:
                twob_remote = True
        ene_list = [None]*self.no**2
        ene_tol = self.ene_tol
        if (len(self.pairlist_close) > 0) and (twob_remote == False):
            t1 = time_now()
            ene_list, ene_close, T_matrix = ene_iteration(self.pairlist_close, "close", T_matrix, ene_list, ene_tol)
            print_time(['OSV-MP2 residue for close pairs', time_elapsed(t1)], log)
        if len(self.pairlist_remote) > 0:
            t1 = time_now()
            ene_list, ene_remote, T_matrix = ene_iteration(self.pairlist_remote, "remote", T_matrix, ene_list, ene_tol)
            print_time(['OSV-MP2 residue for remote pairs', time_elapsed(t1)], log)
        ene_total = ene_close + ene_remote
        print_time(['OSV-MP2 residue', time_elapsed(t0)], log)
        #self.norm = delt
        msg = "\nOSV-MP2 energy converged!"
        if len(self.pairlist_remote) > 0:
            msg += "\nMP2 energy for close pairs:     %.10f"%(ene_close)
            msg += "\nMP2 energy for remote pairs:     %.10f"%(ene_remote)
            msg += "\n                                   -------------"
        msg += "\nOSV-MP2 Converged with the energy: %.10f"%ene_total
        log.info(msg)
        print_time(['OSV-MP2', time_elapsed(tt)], log)

        self.t_mp2_e = time_now() - t_mp2_e
        if (self.use_mbe):
            '''if 1700 in self.pairlist and ene_list[1700] is not None:
                print(len(self.pairlist), '%.7E '%ene_list[1700])#''.join(['%.4E '%ene_list[ipair] for ipair in self.pairlist_close]))'''
            if self.clus_type == 21:
                ene_total = ene_remote
        else:
            self.x_ii = None
            self.emu_i = None
            self.X_matrix = None
            self.emu_ij = None
            self.K_matrix = None
        if (self.use_mbe):
            return ene_list, ene_total, T_matrix
        else:
            if (self.grad_cal):
                return ene_total, T_matrix
            else:
                return ene_total


