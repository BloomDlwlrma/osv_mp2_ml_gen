import os
import sys
import numpy as np
from osvmp2 import ZCPL
from osvmp2.OSVL import get_ene
from pyscf import lib
from osvmp2.osvutil import *
from osvmp2.ga_addons import *
import time
import itertools
#from itertools import imap
import h5py
from mpi4py import MPI
import gc

#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm

def two_body_l(self, oneb_mo):
    def get_cluslist(pairlist, if_remote=False):
        def get_tmatm(i, j):
            tdim_i = self.nosv[i]
            tdim_j = self.nosv[j]
            if if_remote:
                return tdim_i*tdim_j
            else:
                return sum([tdim_i*tdim_i, tdim_i*tdim_j, tdim_j*tdim_j])
            #return np.mean([tdim_i, tdim_j])
        nosv_list = []
        cluslist = []
        allpairs = []
        for ipair in pairlist:
            i = ipair//self.no
            j = ipair%self.no
            if i != j:
                nosv_list.append(get_tmatm(i, j))
                pairs = [i*self.no+i, ipair, j*self.no+j]
                cluslist.append(pairs)
                allpairs.extend(pairs)
        cluslist = [clus_i for nosv_i, clus_i in sorted(zip(nosv_list, cluslist), reverse=True)]
        
        return cluslist, allpairs
    all_two = []
    #twob_pair_remote = []
    #twob_pair_close = []
    all_two_corr = []
    #idx_list = range(len(oneb_mo))
    #count = 0
    twob_pair_close, all_two = get_cluslist(self.refer_pairlist_close)
    twob_pair_remote, all_two_corr = get_cluslist(self.refer_pairlist_remote, if_remote=True)
    return twob_pair_remote, twob_pair_close, sorted(all_two)#, sorted(all_two_corr)

def three_body(self, oneb_mo, log):
    def check_remote(i, j, k):
        ij = i*self.no+j
        ik = i*self.no+k
        jk = j*self.no+k
        if_remote = False
        for ipair in [ij, ik, jk]:
            if (self.if_remote[ipair]) or (self.s_ratio[ipair] < self.disc_tol):
                if_remote = True
                break
        if if_remote == False:
            s_list = sorted([self.s_ratio[ij], self.s_ratio[ik], self.s_ratio[jk]])
            s_mean = np.mean(s_list)
            #s_mean = sorted(s_list)[1]
            if (s_mean < self.threeb_tol) and (s_list[1] < self.threeb_tol):
                if_remote = True
        return if_remote
            
    def get_pairs(i, j, k):
        i, j, k = sorted([i, j, k])
        return [i*self.no+i, i*self.no+j, i*self.no+k, j*self.no+j, j*self.no+k, k*self.no+k]

    def get_tmatm(i, j, k):
        tdim_i = self.nosv[i]
        tdim_j = self.nosv[j]
        tdim_k = self.nosv[k]
        return sum([tdim_i*tdim_i, tdim_i*tdim_j, tdim_i*tdim_k, 
                        tdim_j*tdim_j, tdim_j*tdim_k, tdim_k*tdim_k])

    self.n3b = 0
    nosv_3b = []
    allpairs_3bb = []
    threeb_pair = []
    for i, j, k in itertools.combinations(self.refer_molist, 3):
        if check_remote(i, j, k) is False:
            nosv_3b.append(get_tmatm(i, j, k))
            pairs_3b = get_pairs(i, j, k)
            allpairs_3bb.extend(pairs_3b)
            threeb_pair.append(pairs_3b)
        self.n3b += 1
    threeb_pair = [clus_i for nosv_i, clus_i in sorted(zip(nosv_3b, threeb_pair), reverse=True)]
    return threeb_pair, sorted(allpairs_3bb)

def clus_selection(self, log):
    def get_seg(self):
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
            #return get_slice(range(nrank), job_list=cluslist)
        def merge_segs(list0, list1):
            list_01 = []
            for idx, (seg0, seg1) in enumerate(zip(list0, list1)):
                if (seg0 is not None) and (seg1 is not None):
                    list_01.append(seg0 + seg1)
                elif seg0 is None:
                    list_01.append(seg1)
                elif seg1 is None:
                    list_01.append(seg0)
            return list_01
        def get_nclus(iclus):
            if iclus is None:
                return 0
            else:
                return len(iclus)
        self.oneblist_all = get_si(self.oneb_pair)
        self.oneblist_rank = self.oneblist_all[irank]
        self.n1b_rank = get_nclus(self.oneblist_rank)
        
        self.refer_pairs_remote = self.pairlist_remote[:]
        self.twoblist_remote_all = get_si(self.twob_remote_pair)
        self.twoblist_close_all = get_si(self.twob_close_pair)
        #self.twoblist_all = get_si(self.twob_close_pair + self.twob_remote_pair)
        self.twoblist_all = merge_segs(self.twoblist_remote_all, self.twoblist_close_all)
        self.twoblist_close_rank = self.twoblist_close_all[irank]
        self.twoblist_remote_rank = self.twoblist_remote_all[irank]
        self.twoblist_rank = self.twoblist_all[irank]
        self.n2bc_rank = get_nclus(self.twoblist_close_rank)
        self.n2br_rank = get_nclus(self.twoblist_remote_rank)
        self.n2b_rank = get_nclus(self.twoblist_rank)

        if self.threeb_tol != 1:    
            self.threeblist_all = get_si(self.threeb_pair)
            self.cluslist_3 = self.threeblist_all[irank]
        else:
            self.threeblist_all, self.cluslist_3 = None, None
        self.nclus3_rank = get_nclus(self.cluslist_3)

        #For synchronizations in batches
        def cluslist_syn(clus_all, clus_rank):
            if clus_all is None:
                clus_rank = None
            else:
                nclus_list = [0]*nrank
                for rank_i, clus_i in enumerate(clus_all):
                    if clus_i is not None:
                        nclus_list[rank_i] = len(clus_i)
                max_num = max(nclus_list)
                nclus_list = nclus_list[irank]
                if nclus_list < max_num:
                    if clus_rank is None:
                        clus_rank = [None]*max_num
                    else:
                        clus_rank = clus_rank + [None]*(max_num-nclus_list)
            return clus_rank
        self.oneblist_rank = cluslist_syn(self.oneblist_all, self.oneblist_rank)
        self.twoblist_close_rank = cluslist_syn(self.twoblist_close_all, self.twoblist_close_rank)
        self.twoblist_remote_rank = cluslist_syn(self.twoblist_remote_all, self.twoblist_remote_rank)
        self.cluslist_3 = cluslist_syn(self.threeblist_all, self.cluslist_3)
        if self.grad_cal:
            self.twoblist_rank = cluslist_syn(self.twoblist_all, self.twoblist_rank)

    def mbe_info(log):
        msg_list = [['Number of 1-Body clusters', len(self.mo_list)]]
        print_align(msg_list, align='lr', indent=4, log=log)
        log.info('    ---------------------------------------------')
        msg_list = [['2B selection threshold', '%.1E'%self.remo_tol],
                        ['Number of 2-Body clusters', self.n2b_close+self.n2b_remote],
                        ['Number of close 2-Body clusters', self.n2b_close],
                        ['Number of remote 2-Body clusters', self.n2b_remote]]
        print_align(msg_list, align='lr', indent=4, log=log)
        if self.threeb_tol != 1:
            log.info('    -----------------------------------------------')
            msg_list = [['3B selection threshold', '%.1E'%self.threeb_tol],
                        ['Number of 3-Body clusters', self.n3b_close],
                        ['Number of remote 3-Body clusters', self.n3b - self.n3b_close]]
            print_align(msg_list, align='lr', indent=4, log=log)
            log.info('    -----------------------------------------------')
    
    t0 = time_now()
    #Determination of two-body clusters
    mo_list = self.mo_list#range(self.no)
    oneb_mo = [[i] for i in mo_list]
    self.oneb_pair = [[i*self.no+i] for i in mo_list]
    
    #Determination of two-body clusters
    self.twob_remote_pair, self.twob_close_pair, self.allpairs_2b = two_body_l(self, oneb_mo)
    self.n2b_close = len(self.twob_close_pair)
    self.n2b_remote = len(self.twob_remote_pair)
    
    #Determination of three-body clusters
    if self.threeb_tol != 1:
        self.threeb_pair, self.allpairs_3b = three_body(self, oneb_mo, log)
        self.n3b_close = len(self.threeb_pair)
    else:
        self.n3b_close = 0
        self.threeb_pair, self.allpairs_3b = [], []
    
    #Set up counter for accumulation of MBE values
    self.count_2b = [0]*self.no**2
    self.count_3b = [0]*self.no**2
    for ipair in self.allpairs_2b:
        self.count_2b[ipair] += 1
    for ipair in self.allpairs_3b:
        self.count_3b[ipair] += 1
        i = ipair//self.no
        j = ipair%self.no
        if i != j:
            for pair_i in [i*self.no+i, j*self.no+j]:
                self.count_3b[pair_i] -= 1
    get_seg(self)
    mbe_info(log)
    if irank == 0:
        print_time(['OSV cluster selection', time_elapsed(t0)], log)


def get_buffers(self):
    def get_tmat_buffer():
        def get_pairslice(self, if_sort=True):
            if if_sort:
                def get_slice(pair_list):
                    job_size = len(pair_list)
                    rank_list = []
                    for rank_i, node_i in itertools.product(range(nrank//self.nnode), range(self.nnode)):
                        rank_list.append(self.rank_slice[node_i][rank_i])
                    job_slice = [None]*nrank
                    for idx, ipair in enumerate(pair_list):
                        rank_idx = rank_list[idx%nrank]
                        if job_slice[rank_idx] == None:
                            job_slice[rank_idx] = [ipair]
                        else:
                            job_slice[rank_idx].append(ipair)
                    return job_slice
                def merge_segs(list0, list1):
                    list_01 = []
                    for idx, (seg0, seg1) in enumerate(zip(list0, list1)):
                        if (seg0 is not None) and (seg1 is not None):
                            list_01.append(seg0 + seg1)
                        elif seg0 is None:
                            list_01.append(seg1)
                        elif seg1 is None:
                            list_01.append(seg0)
                    return list_01
                def sort_pairlist(pair_list):
                    cost_list = []
                    cost_record = [None]*self.no**2
                    for ipair in pair_list:
                        i = ipair//self.no
                        j = ipair%self.no
                        if self.if_remote[ipair]:
                            cost_ij = 2*self.nosv[i]*self.nosv[j]*self.nosv[j]
                            cost_ij += 2*self.nosv[i]*self.nosv[i]*self.nosv[j]
                        else:
                            nosv_ij = self.nosv[i]+self.nosv[j]
                            for k in self.k_list[ipair]:
                                    nosv_ik = self.nosv[i]+self.nosv[k]
                                    nosv_kj = self.nosv[j]+self.nosv[k]
                                    cost_ij = nosv_ij*nosv_ik*nosv_ik + nosv_ij*nosv_ik*nosv_ij
                                    cost_ij += nosv_ij*nosv_kj*nosv_kj + nosv_ij*nosv_kj*nosv_ij
                        cost_list.append(cost_ij)
                        cost_record[ipair] = cost_ij
                    '''if self.if_remote[pair_list[0]] is False:
                        for ipair in pair_list:
                            i = ipair//self.no
                            j = ipair%self.no
                            kcost_list = []
                            for k in self.k_list[ipair]:
                                cost_k = cost_record[get_pair(i,k)] + cost_record[get_pair(j,k)]
                                kcost_list.append(cost_k)
                            self.k_list[ipair] = [k for cot_k, k in sorted(zip(kcost_list, self.k_list[ipair]))]'''
                    pairs_sorted = [ipair for cost_i, ipair in sorted(zip(cost_list, pair_list))]#, reverse=True)]
                    return pairs_sorted
                pair_slice_close = get_slice(sort_pairlist(self.refer_pairlist_close))
                pair_slice_remote = get_slice(sort_pairlist(self.refer_pairlist_remote))
                pair_slice = merge_segs(pair_slice_close, pair_slice_remote)
            else:
                pair_slice_close = get_slice(range(nrank), job_list=self.refer_pairlist_close)
                pair_slice_remote = get_slice(range(nrank), job_list=self.refer_pairlist_remote)
                pair_slice = []
                for rank_i in range(nrank):
                    if (pair_slice_remote[rank_i] == None) and (pair_slice_close[rank_i] == None):
                        pair_slice.append(None)
                    else:
                        slice_i = []
                        if pair_slice_remote[rank_i] != None:
                            slice_i.extend(pair_slice_remote[rank_i])
                        if pair_slice_close[rank_i] != None:
                            slice_i.extend(pair_slice_close[rank_i])
                        pair_slice.append(sorted(slice_i))
            return pair_slice_close, pair_slice_remote, pair_slice
        def get_pair(i, j):
            if i < j:
                return i*self.no+j
            else:
                return j*self.no+i
        def get_klist():
            def get_k(i, j):
                def check_redundant(self, i, j, k):
                    if (i == k) or (j == k):
                        return False
                    else:
                        if_red = False
                        pairs = [get_pair(i, k), get_pair(j, k)]
                        for ipair in pairs:
                            if self.if_remote[ipair] or self.if_discarded[ipair]:
                                if_red = True
                                break
                        if if_red is False:
                            if i != j:
                                s_list = sorted([self.s_ratio[get_pair(i, j)], self.s_ratio[get_pair(i, k)], 
                                                    self.s_ratio[get_pair(j, k)]])
                                s_mean = np.mean(s_list)
                                
                                if (s_mean < self.threeb_tol) and (s_list[1] < self.threeb_tol):
                                    if_red = True
                        return if_red
                k_list = []
                for k in self.refer_molist:
                    if check_redundant(self, i, j, k) is False:
                        k_list.append(k)
                return k_list
            klist = [None]*self.no**2
            for ipair in self.refer_pairlist_close:
                klist[ipair] = get_k(ipair//self.no, ipair%self.no)
            for ipair in self.refer_pairlist_remote:
                klist[ipair] = [ipair//self.no, ipair%self.no]
            return klist
        
        self.k_list = get_klist()
        '''if self.mbe_mode == 1:
            if_sort = False
        else:'''
        if_sort = True
        self.pairslice_close_tmat, self.pairslice_remote_tmat, self.pairslice_all_tmat = get_pairslice(self, if_sort)
        self.tmat_address = [None]*self.no**2
        self.dim_tmat = [None]*self.no**2
        size_tmat = 0
        for rank_i, pair_i in enumerate(self.pairslice_all_tmat):
            if pair_i is not None:
                t_idx0 = 0
                for ipair in sorted(pair_i):
                    i = ipair//self.no
                    j = ipair%self.no
                    if (self.if_remote[ipair]):
                        t_idx1 = t_idx0 + self.nosv[i]*self.nosv[j]
                        self.dim_tmat[ipair] = (self.nosv[i], self.nosv[j])
                    else:
                        t_idx1 = t_idx0 + (self.nosv[i]+self.nosv[j])*(self.nosv[i]+self.nosv[j])
                        self.dim_tmat[ipair] = (self.nosv[i]+self.nosv[j], self.nosv[i]+self.nosv[j])
                    self.tmat_address[ipair] = [rank_i, [t_idx0, t_idx1]]
                    if rank_i == irank: 
                        size_tmat += t_idx1 - t_idx0
                    t_idx0 = t_idx1
        if self.pairslice_all_tmat[irank] is not None:
            self.tmat_ga = np.zeros(size_tmat)
            self.tmat12_ga = np.zeros(size_tmat)
        else:
            self.tmat_ga = None
            self.tmat12_ga = None
        self.win_tmat = create_win(self.tmat_ga, comm=comm)
        self.win_tmat12 = create_win(self.tmat12_ga, comm=comm)
        
    def get_mbe_buffers():
        #Save MO_basis variables
        self.win_ene, self.mp2e_node = get_shared(6, dtype='f8')
        self.mp2e_rank = np.zeros(6, dtype='f8')
        if (self.grad_cal):
            mo_slice = get_slice(range(nrank), job_list=self.mo_list)
            self.gamma_address = [None]*self.no
            self.cpn_address = [None]*self.no
            self.dim_cpn = [None]*self.no
            len_cpn = 0
            for rank_i, mo_i in enumerate(mo_slice):
                if mo_i != None:
                    g_idx0, n_idx0 = 0, 0
                    for num in mo_i:
                        g_idx1 = g_idx0 + 1
                        self.gamma_address[num] = [rank_i, [g_idx0, g_idx1]]
                        n_idx1 = n_idx0 + self.nosv_cp[num]*self.nosv[num]
                        self.cpn_address[num] = [rank_i, [n_idx0, n_idx1]]
                        self.dim_cpn[num] = (self.nosv_cp[num], self.nosv[num])
                        if rank_i == irank: len_cpn += (n_idx1 - n_idx0)
                        g_idx0, n_idx0 = g_idx1, n_idx1
            mo_slice = mo_slice[irank]
            if (mo_slice != None) and (self.use_cposv):
                self.cpn_ga = np.zeros(len_cpn, dtype='f8')
            else:
                self.gamma_mo, self.cpn_ga = None, None
                        
            self.win_cpn = create_win(self.cpn_ga, comm=comm)
            self.win_dmp2, self.DMP2 = get_shared((self.nao, self.nao), dtype='f8')
            if irank_shm != 0:
                self.dmp2_save = np.zeros((self.nao, self.nao), dtype='f8')
    def ml_buffers():
        self.pairlist_offdiag = [ipair for ipair in self.refer_pairlist if ipair//self.no != ipair%self.no]
        self.idx_pair_off = [None]*self.no**2
        for idx, ipair in enumerate(self.pairlist_offdiag):
            self.idx_pair_off[ipair] = idx
        self.idx_pair = [None]*self.no**2
        for idx, ipair in enumerate(self.refer_pairlist):
            self.idx_pair[ipair] = idx
        self.idx_3b0 = 0
        for rank_i, threeb_rank in enumerate(self.threeblist_all):
            if threeb_rank is None: break
            self.idx_3b1 = self.idx_3b0 + len(threeb_rank)
            if rank_i == irank:
                break
            else:
                self.idx_3b0 = self.idx_3b1
        self.idx_mo = [None]*self.no
        for idx_i, i in enumerate(self.mo_list):
            self.idx_mo[i] = idx_i
        self.win_ene_1b2b, self.ene_1b2b = get_shared(self.no**2)
        self.win_ene_1b_all, self.ene_1b_all = get_shared((len(self.mo_list), 3))
        self.win_ene_2b_all, self.ene_2b_all = get_shared((len(self.pairlist_offdiag), 3))
        self.win_ene_3b_all, self.ene_3b_all = get_shared((len(self.threeb_pair), 7))
        self.ene_3b_rank = []
        if self.mbe_mode == 0 or self.mbe_mode == 1:
            #Coulums: IC1, IC2, IC3, CT2, CT3, DC2, DC3, EC2, EC3
            self.win_ene_decom_mbe, self.ene_decom_mbe = get_shared((len(self.refer_pairlist), 9))
            self.ene_decom_mbe_rank = np.zeros_like(self.ene_decom_mbe)
        if self.mbe_mode == 0 or self.mbe_mode == 2:
            #Coulums: IC, CT, DT, EX
            self.win_ene_decom_nonmbe, self.ene_decom_nonmbe = get_shared((len(self.refer_pairlist), 4))
    def get_mos(iclus):
        mo_list = []
        for ipair in iclus:
            i = ipair//self.no
            j = ipair%self.no
            mo_list.extend([i, j])
        return sorted(list(set(mo_list)))
    def get_pairs_full(iclus):
        pairs_full = []
        for ipair in iclus:
            pairs_full.append(ipair)
            i = ipair//self.no
            j = ipair%self.no
            if i != j:
                pairs_full.append(j*self.no+i)
        return pairs_full
    def init_buffer(dim_mat):
        if dim_mat == []:
            return None
        else:
            return np.empty(max(dim_mat), dtype='f8')
    if (self.n1b_rank + self.n2b_rank + self.nclus3_rank) > 0:
        if self.n2br_rank > 0:
            dim_xii = []
            dim_emui = []
            for iclus in self.twoblist_remote_rank:
                if iclus is None: break
                for ipair in iclus:
                    if self.if_remote[ipair]:
                        i = ipair//self.no
                        j = ipair%self.no
                        dim_xii.append(get_buff_size(self.dim_xii, [i, j]))
                        dim_emui.append(get_buff_size(self.dim_emui, [i, j]))
            self.buffer_xii = init_buffer(dim_xii)
            self.buffer_emui = init_buffer(dim_emui)
        if self.grad_cal:
            pair_slice = get_pairslice(self, if_remote=False, if_full=True)
            self.dim_sf_cp = [None]*self.no**2
            for pair_i in pair_slice:
                if pair_i is not None:
                    for ipair in pair_i:
                        i = ipair // self.no
                        j = ipair % self.no
                        self.dim_sf_cp[ipair] = (self.nosv_cp[i], self.nosv[j])
        dim_kmat = []
        dim_sf = []
        dim_xmat = []
        dim_emuij = []
        dim_qmat = []
        dim_ti = []
        dim_qcp = []
        dim_sf_cp = []
        for cluslist in [self.cluslist_3, self.twoblist_rank, self.oneblist_rank]:
            if cluslist is not None:
                for iclus in cluslist:
                    if iclus is None: break
                    dim_kmat.append(get_buff_size(self.dim_kmat, iclus))
                    dim_sf.append(get_buff_size(self.dim_sf, iclus))
                    molist = get_mos(iclus)
                    dim_qmat.append(get_buff_size(self.dim_qmat, molist))
                    dim_ti.append(len(iclus))
                    dim_qcp.append(len(iclus))
                    pairs_remote = []
                    pairs_close = []
                    for ipair in iclus:
                        if self.if_remote[ipair]:
                            pairs_remote.append(ipair)
                        else:
                            pairs_close.append(ipair)
                    dim_xmat.append(get_buff_size(self.dim_xmat, pairs_close))
                    dim_emuij.append(get_buff_size(self.dim_emuij, pairs_close))

                    twob_remote = False
                    if self.lg_dr:
                        pairlist = iclus
                    else:
                        pairlist = pairs_close
                        if len(iclus)==3:
                            if self.if_remote[iclus[1]]:
                                twob_remote = True
                    if (self.grad_cal) and (twob_remote == False):
                        pairs_full = get_pairs_full(pairlist)
                        dim_sf_cp.append(get_buff_size(self.dim_sf_cp, pairs_full))

        self.buffer_kmat = init_buffer(dim_kmat)
        self.buffer_smat = init_buffer(dim_sf)
        self.buffer_fmat = init_buffer(dim_sf)
        self.buffer_xmat = init_buffer(dim_xmat)
        self.buffer_emuij = init_buffer(dim_emuij)
        self.buffer_tmat = init_buffer(dim_kmat)
        if (self.grad_cal):
            #self.buffer_tmat = init_buffer(dim_kmat)
            self.buffer_qmat = init_buffer(dim_qmat)
            self.buffer_scp = init_buffer(dim_sf_cp)
            self.buffer_fcp = init_buffer(dim_sf_cp)
    self.x_ii = [None]*self.no
    self.emu_i = [None]*self.no
    self.K_matrix = [None]*self.no**2
    self.S_matrix = [None]*self.no**2
    self.F_matrix = [None]*self.no**2
    self.X_matrix = [None]*self.no**2
    self.emu_ij = [None]*self.no**2
    
    #for gradient
    self.T_matrix = [None]*self.no**2
    self.Q_matrix = [None]*self.no
    self.Ti = [None]*self.no
    self.Q_matrix_cp = [None]*self.no
    self.S_matrix_cp = [None]*self.no**2
    self.F_matrix_cp = [None]*self.no**2

    get_tmat_buffer()
    get_mbe_buffers()        
    if self.ml_test:
        ml_buffers()

def buffer_release(self, cal_type):
    #Free windows
    if cal_type == 'e':
        win_list = [self.win_xmat, self.win_emuij, self.win_kmat]
        if len(self.refer_pairs_remote) > 0:
            win_list += [self.win_xii, self.win_emui]
        for win in win_list:
            free_win(win)
        #clear GAs
        self.xii_ga = None
        self.emui_ga = None
        self.xmat_ga = None
        self.emuij_ga = None
        self.kmat_ga = None
        
        #Clear variables
        self.buffer_kmat = None
        self.buffer_xmat = None
        self.buffer_emuij = None
        self.buffer_xii = None
        self.buffer_emui = None

        self.K_matrix = None
        self.X_matrix = None
        self.emu_ij = None
        self.x_ii = None
        self.emu_i = None
        
    if self.grad_cal and cal_type in 'dr':
        win_list = [self.win_cpn, self.win_smat, self.win_fmat]
        if (self.direct_int is False) and (self.use_cposv):
            win_list += [self.win_scp, self.win_fcp]
        for win in win_list:
            free_win(win)
    
        #clear GAs
        self.smat_ga = None
        self.fmat_ga = None
        self.scp_ga = None
        self.fcp_ga = None

        #Clear variables
        self.buffer_tmat = None
        self.buffer_qmat = None
        self.buffer_smat = None
        self.buffer_fmat = None
        self.buffer_scp = None
        self.buffer_fcp = None

        self.T_matrix = None
        self.Q_matrix = None
        self.S_matrix = None
        self.F_matrix = None
        self.S_matrix_cp = None
        self.F_matrix_cp = None

    elif cal_type == 'gamma':
        win_list = [self.win_tmat, self.win_qmat]
        if self.direct_int == False:
            win_list.append(self.win_ialp_mo)
            if len(self.refer_pairs_remote) > 0:
                win_list.append(self.win_imup)
        for win in win_list:
            free_win(win)
        self.tmat_ga = None
        self.qmat_ga = None
        self.cpn_ga = None
        self.imup_ga = None
        self.ti_ga = None
        self.qcp_ga = None
        self.ialp_mo = None
    gc.collect()

def get_eg(self, cluslist, clus_type, cal_type, log):
    def read_ga(cal_type):
        def read_sf(read_f=True):
            self.buffer_smat, self.S_matrix = read_GA(self.sf_address, self.pairlist, self.buffer_smat, self.win_smat, dtype='f8', list_col=self.S_matrix, dim_list=self.dim_sf)
            if read_f:
                self.buffer_fmat, self.F_matrix = read_GA(self.sf_address, self.pairlist, self.buffer_fmat, self.win_fmat, dtype='f8', list_col=self.F_matrix, dim_list=self.dim_sf)
            for ipair in self.pairlist:
                i = ipair // self.no
                j = ipair % self.no
                if i != j:
                    pair_ji = j*self.no+i
                    self.S_matrix[pair_ji] = self.S_matrix[ipair].T
                    if read_f:
                        self.F_matrix[pair_ji] = self.F_matrix[ipair].T
        
        if cal_type == 'e':
            if iclus is not None:
                t1 = time_now()
                if len(self.pairlist_remote) > 0:
                    self.buffer_xii, self.x_ii = read_GA(self.xii_address, self.mo_remote, self.buffer_xii, self.win_xii, dtype='f8', list_col=self.x_ii, dim_list=self.dim_xii)
                    self.buffer_emui, self.emu_i = read_GA(self.emui_address, self.mo_remote, self.buffer_emui, self.win_emui, dtype='f8', list_col=self.emu_i, dim_list=self.dim_emui)
                self.buffer_xmat, self.X_matrix = read_GA(self.xmat_address, self.pairlist_close, self.buffer_xmat, self.win_xmat, dtype='f8', list_col=self.X_matrix, dim_list=self.dim_xmat)
                self.buffer_emuij, self.emu_ij = read_GA(self.emuij_address, self.pairlist_close, self.buffer_emuij, self.win_emuij, dtype='f8', list_col=self.emu_ij, dim_list=self.dim_emuij)
                self.buffer_kmat, self.K_matrix = read_GA(self.kmat_address, self.pairlist, self.buffer_kmat, self.win_kmat, dtype='f8', list_col=self.K_matrix, dim_list=self.dim_kmat)
                read_sf()
                if (self.clus_type == 3):
                    self.buffer_tmat, self.T_matrix = read_GA(self.tmat_address, self.pairlist, self.buffer_tmat, self.win_tmat12, dtype='f8', list_col=self.T_matrix, dim_list=self.dim_tmat)
                elif (self.clus_type == 20) or (self.clus_type == 21):
                    ii, ij, jj = self.pairlist
                    self.buffer_tmat, self.T_matrix = read_GA(self.tmat_address, [ii, jj], self.buffer_tmat, self.win_tmat12, dtype='f8', list_col=self.T_matrix, dim_list=self.dim_tmat)
                self.t_read += time_now() - t1
            comm.Barrier()
        else:
            if iclus is not None:
                t1 = time_now()
                self.buffer_tmat, self.T_matrix = read_GA(self.tmat_address, self.pairlist, self.buffer_tmat, self.win_tmat, dtype='f8', list_col=self.T_matrix, dim_list=self.dim_tmat)
                if 'r' not in cal_type:
                    read_sf(read_f=False)
                else:
                    read_sf(read_f=True)
                if (self.clus_type != 3) and 'd' in cal_type:
                    self.buffer_qmat, self.Q_matrix = read_GA(self.qmat_address, self.mo_list, self.buffer_qmat, self.win_qmat, dtype='f8', list_col=self.Q_matrix, dim_list=self.dim_qmat)
                if (self.clus_type != 21 or self.lg_dr) and (self.direct_int is False) and (self.use_cposv) and ('r' in cal_type):
                        self.buffer_scp, self.S_matrix_cp = read_GA(self.sf_cp_address, self.pairs_full, self.buffer_scp, self.win_scp, dtype='f8', list_col=self.S_matrix_cp, dim_list=self.dim_sf_cp)
                        self.buffer_fcp, self.F_matrix_cp = read_GA(self.sf_cp_address, self.pairs_full, self.buffer_fcp, self.win_fcp, dtype='f8', list_col=self.F_matrix_cp, dim_list=self.dim_sf_cp)
                self.t_read += time_now() - t1
            comm.Barrier()
            
            t1 = time_now()
            if (iclus is not None) and (self.direct_int) and (self.use_cposv) and ('r' in cal_type):
                if (self.clus_type != 21) or (self.lg_dr):
                    buf_idx0 = 0
                    for ipair in self.pairs_full:
                        dim_i = self.dim_sf_cp[ipair]
                        buf_idx1 = buf_idx0 + np.product(dim_i)
                        self.S_matrix_cp[ipair] = self.buffer_scp[buf_idx0:buf_idx1]
                        self.F_matrix_cp[ipair] = self.buffer_fcp[buf_idx0:buf_idx1]
                        rank_i, idx_list = self.sf_cp_address[ipair]
                        idx0, idx1 = idx_list
                        with h5py.File("%s/sf_cp_%d.tmp"%(self.dir_sf_cp, rank_i), 'r') as file_sf_cp:
                            file_sf_cp['scp'].read_direct(self.S_matrix_cp[ipair], source_sel=np.s_[idx0:idx1])
                            file_sf_cp['fcp'].read_direct(self.F_matrix_cp[ipair], source_sel=np.s_[idx0:idx1])
                        self.S_matrix_cp[ipair] = self.S_matrix_cp[ipair].reshape(dim_i)
                        self.F_matrix_cp[ipair] = self.F_matrix_cp[ipair].reshape(dim_i)
                        buf_idx0 = buf_idx1
            self.t_read += time_now() - t1

    def clear_ga(iclus, clus_next, cal_type):
        if (iclus is not None):
            if (clus_next is None):
                mo_diff = self.mo_list
                pair_diff = iclus
            else:
                mo_next = []
                for ipair in clus_next:
                    mo_next.extend([ipair//self.no, ipair%self.no])
                mo_next = sorted(list(set(mo_next)))
                mo_diff = self.mo_list[:]
                for i, j in zip(self.mo_list, mo_next):
                    if i != j: break
                    mo_diff.remove(i)
                pair_diff = iclus[:]
                for ipair, jpair in zip(iclus, clus_next):
                    if ipair != jpair: break
                    pair_diff.remove(ipair)

            for i in mo_diff:
                if cal_type == 'e':
                    self.x_ii[i] = None
                    self.emu_i[i] = None
                if self.grad_cal and 'r' in cal_type:
                    self.Q_matrix[i] = None
            
            for ipair in pair_diff:
                if cal_type == 'e':
                    self.X_matrix[ipair] = None
                    self.emu_ij[ipair] = None
                    self.K_matrix[ipair] = None
                    self.T_matrix[ipair] = None
                else:
                    self.T_matrix[ipair] = None
                i = ipair//self.no
                j = ipair%self.no
                for pair_i in [ipair, j*self.no+i]:
                    self.S_matrix[pair_i] = None
                    self.F_matrix[pair_i] = None
                    self.S_matrix_cp[pair_i] = None
                    self.F_matrix_cp[pair_i] = None


    def kernel(iclus, clus_next, cal_type):
        self.cal_type = cal_type
        if iclus is not None:
            self.pairlist = iclus
            self.mo_list = []
            self.mo_remote = []
            self.pairlist_ij = []
            self.pairlist_remote = []
            self.pairlist_close = []
            self.pairidx = [None]*self.no**2
            self.pairs_full = []
            #if self.pairlist != []:
            for idx, ipair in enumerate(self.pairlist):
                i = ipair//self.no
                j = ipair%self.no
                if i != j:
                    self.pairlist_ij.append(ipair)
                self.pairidx[ipair] = idx
                self.mo_list.extend([i, j])
                if (self.if_remote[ipair]):
                    self.pairlist_remote.append(ipair)
                    self.mo_remote.extend([i, j])
                else:
                    self.pairlist_close.append(ipair)
            if (self.grad_cal):
                if self.lg_dr:
                    pairlist = self.pairlist
                else:
                    pairlist = self.pairlist_close
                for ipair in pairlist:
                    i = ipair//self.no
                    j = ipair%self.no
                    self.pairs_full.append(ipair)
                    if i != j:
                        self.pairs_full.append(j*self.no+i)

            self.mo_list = sorted(list(set(self.mo_list)))
            self.mo_remote = sorted(list(set(self.mo_remote)))

            if len(iclus) == 1:
                self.clus_type = 1
            elif len(iclus) == 3:
                if self.if_remote[iclus[1]]:
                    self.clus_type = 21
                else:
                    self.clus_type = 20
            else:
                self.clus_type = 3
        #Read variables required
        
        read_ga(cal_type)
        if iclus is not None:
            e_list, e_f, tmat, cpn, dmp2 = None, None, None, None, None
            if cal_type == 'e':
                self.nite_list = []
                #Compute mp2 energy
                t1 = time_now()
                e_list, e_f, tmat = get_ene(self)
                self.t_cal += time_now() - t1
                if self.ml_test:
                    #e_pairs = [e for e in e_list if e is not None]
                    if clus_type in [20, 21]:
                        ii, ij, jj = iclus
                        if clus_type == 21:
                            e2b = e_f + self.ene_1b2b[ii] + self.ene_1b2b[jj]
                        else:
                            e2b = e_f
                        self.ene_1b2b[ij] = e2b
                        self.ene_2b_all[self.idx_pair_off[iclus[1]]] = [self.ene_1b2b[ii], self.ene_1b2b[jj], e2b]#e_pairs
                    elif clus_type == 3:
                        ii, ij, ik, jj, jk, kk = iclus
                        epairs = [self.ene_1b2b[ipair] for ipair in [ii, jj, kk, ij, ik, jk]]
                        self.ene_3b_rank.append(epairs + [e_f])
                    elif clus_type == 1:
                        ii = iclus[0]
                        self.ene_1b2b[ii] = e_f
                        self.ene_1b_all[self.idx_mo[ii//self.no]] = e_f
            if self.grad_cal and cal_type in 'dr':
                #Compute mp2 gradient
                t1 = time_now()
                if (self.grad_cal):
                    tbar, cpn, dmp2 = ZCPL.MO_basis_A(self, self.T_matrix)
                self.t_cal += time_now() - t1
        
        comm.Barrier()
        
        if iclus is not None:
            t1 = time_now()
            save_mo_GA(self, iclus, e_list, e_f, tmat, cpn, dmp2)
            
        #if (self.grad_cal):
        if cal_type == 'e':
            if self.clus_type == 1 or self.clus_type == 20:
                self.win_tmat12.Fence()
            if (self.grad_cal):
                self.win_tmat.Fence()
        
        if self.grad_cal and cal_type in 'dr':
            self.win_cpn.Fence()
        clear_ga(iclus, clus_next, cal_type)
        if iclus is not None:
            self.t_col += time_now() - t1
    t_read = self.t_read.copy()
    t_cal = self.t_cal.copy()
    t_col = self.t_col.copy()
    
    refer_verbose = self.verbose
    self.clus_type = clus_type
    self.verbose = 0
    for idx, iclus in enumerate(cluslist):
        if idx+1 == len(cluslist):
            clus_next = None
        else:
            clus_next = cluslist[idx+1]
        kernel(iclus, clus_next, cal_type)
    self.verbose = refer_verbose
    if irank == 0:
        if self.clus_type==20 or self.clus_type==21:
            if self.clus_type==21 and self.cal_type == 'e':
                clus_type = 'Remote 2'
            else:
                clus_type = 2
        else:
            clus_type = self.clus_type
            
        msg_list = ['    %s-Body time: '%clus_type, 
                        'read: %.2f, '%(self.t_read[1]-t_read[1]),
                        'comp: %.2f, '%(self.t_cal[1]-t_cal[1]),
                        'col: %.2f'%(self.t_col[1]-t_col[1])]
        if self.cal_type == 'e':
            msg_list.append('; ave %d cycles'%np.mean(self.nite_list))
        log.info(''.join(msg_list))

#NEW
def save_mo_GA(self, iclus, e_list, e_f, tmat, cpn, dmp2):
    def accumulate_mo(slice_i, win_i, var, addr, count=1, sup_dim=1, idx_use=False):
        for index, i in enumerate(slice_i):
            rank_i, idx = addr[i]
            idx0, idx1 = idx
            if idx_use:
                i = index
            if count == 1:
                var_i = var[i]
            else:
                var_i = count*var[i]
            win_i.Lock(rank_i)
            win_i.Accumulate(var_i, target_rank=rank_i, target=[idx0*sup_dim*8, (idx1-idx0)*sup_dim, MPI.DOUBLE], op=MPI.SUM)
            win_i.Unlock(rank_i)
    def accu_grad(dmp2, cpn=None, count=1):
        if irank_shm == 0:
            self.DMP2 += dmp2*count
        else:
            self.dmp2_save += dmp2*count
        if (self.use_cposv) and (cpn is not None): 
            accumulate_mo(self.mo_list, self.win_cpn, cpn, self.cpn_address, count=count)
    if self.cal_type == 'e':
        #1-body clus
        if self.clus_type == 1:
            ipair = iclus[0]
            re_count2 = self.count_2b[ipair]
            re_count3 = self.count_3b[ipair]
            reduce_count = (re_count2 + re_count3)
            self.mp2e_rank[1] += e_f
            self.mp2e_rank[2] -= e_f * re_count2
            self.mp2e_rank[3] -= e_f * re_count3
            self.mp2e_rank[0] += e_f * (1-reduce_count)
            #accumulate_mo(self.pairlist, self.win_tmat12, tmat, self.tmat_address, count=(1-re_count2))
            accumulate_mo(self.pairlist, self.win_tmat12, tmat, self.tmat_address)
            if (self.grad_cal) or (self.mbe_mode != 1) or (self.ml_test): 
                accumulate_mo(self.pairlist, self.win_tmat, tmat, self.tmat_address, count=(1-reduce_count))
        #Close 2-body clus
        elif self.clus_type == 20:
            ii, ij, jj = iclus
            self.mp2e_rank[2] += e_f
            re_count3 = self.count_3b[ij]
            self.mp2e_rank[0] += e_f * (1-re_count3)
            self.mp2e_rank[3] -= e_f * re_count3 
            if self.mbe_mode != 2:
                #accumulate_mo(self.pairlist, self.win_tmat12, tmat, self.tmat_address)
                accumulate_mo([ij], self.win_tmat12, tmat, self.tmat_address)
            if (self.grad_cal) or (self.mbe_mode != 1) or (self.ml_test): 
                accumulate_mo(self.pairlist, self.win_tmat, tmat, self.tmat_address, count=(1-re_count3))
        #Remote 2-body clus
        elif self.clus_type == 21:
            self.mp2e_rank[0] += e_f
            self.mp2e_rank[4] += e_f
            if (self.grad_cal) or (self.ml_test): 
                accumulate_mo([iclus[1]], self.win_tmat, tmat, self.tmat_address)
        #3-body clus
        elif self.clus_type == 3:
            self.mp2e_rank[0] += e_f
            self.mp2e_rank[3] += e_f
            if (self.grad_cal) or (self.mbe_mode != 1) or (self.ml_test): 
                accumulate_mo(self.pairlist, self.win_tmat, tmat, self.tmat_address)
    else:
        accu_grad(dmp2, cpn)
def save_ml(self):
    def get_coord_list(mol):
        atom_list = []
        coord_list = []
        for atm in range(mol.natm):
            atom_list.append(mol.atom_pure_symbol(atm))
        return np.asarray(atom_list,dtype='S'), mol.atom_coords()*lib.param.BOHR

    def get_atomic_dist(coord_array):
        natm = coord_array.shape[0]
        dist_array = np.zeros((natm, natm))
        for ia, coi in enumerate(coord_array):
            for ja, coj in enumerate(coord_array):
                dist_array[ia, ja] = np.linalg.norm((coi - coj))
        return dist_array
    #Collect S and F matrix
    self.sf_dim_pairlist = [self.dim_sf[ipair] for ipair in self.refer_pairlist]
    size_sf = sum([np.prod(dim_ij) for dim_ij in self.sf_dim_pairlist])
    win_smat, smat_osv = get_shared(size_sf)
    win_fmat, fmat_osv = get_shared(size_sf)
    if self.smat_ga is not None:
        save_idx0 = 0
        for ipair in self.refer_pairlist:
            rank_i, [ga_idx0, ga_idx1] = self.sf_address[ipair]
            save_idx1 = save_idx0 + (ga_idx1 - ga_idx0)
            if rank_i == irank:
                smat_osv[save_idx0:save_idx1] = self.smat_ga[ga_idx0:ga_idx1]
                fmat_osv[save_idx0:save_idx1] = self.fmat_ga[ga_idx0:ga_idx1]
            save_idx0 = save_idx1
    if self.mbe_mode != 2:
        if self.ene_3b_rank != []:
            self.ene_3b_all[self.idx_3b0:self.idx_3b1] = np.asarray(self.ene_3b_rank)
    comm.Barrier()
    step = self.mol.name
    if irank == 0:
        def save_file(data_dic, step, data_file="ml_features.hdf5"):
            if os.path.isfile(data_file):
                fmode = "r+"
            else:
                fmode = "w"
            with h5py.File(data_file, fmode) as f:
                for key in data_dic.keys():
                    dat = data_dic[key]
                    if type(dat) not in [type([0]), type(np.arange(0))]:
                        dat = np.asarray([dat])
                    else:
                        dat = np.asarray(dat)
                    f["%s/%s"%(step, key)] = dat
        data = {}
        data["nocc_core"] = self.nocc_core
        data["nocc"] = self.no
        data["mo_list"] = np.arange(self.nocc_core, self.no)
        data["nosv(mo_list)"] = [self.nosv[i] for i in data["mo_list"]]
        data["loc_fock(nocc,nocc)"] = self.loc_fock
        data["Coulomb(pairlist)"] = self.coulomb_pair #
        data["Exchange(pairlist)"] = self.exchange_pair #
        data["Smat_osv(pairlist)"] = smat_osv #
        data["Fmat_osv(pairlist)"] = fmat_osv #
        data["sf_osv_dim(pairlist)"] = [self.dim_sf[ipair] for ipair in self.refer_pairlist]
        data["Kmat_osv(pairlist)"] = self.Kmat_osv #
        data["kmat_osv_dim(pairlist)"] = [self.dim_kmat[ipair] for ipair in self.refer_pairlist]
        data["s_ratio(nocc,nocc)"] = self.s_ratio
        data["pairlist"] = self.refer_pairlist
        data["pairlist_offdiag"] = self.pairlist_offdiag
        data["pairlist_screened"] = self.refer_pairlist_remote
        if self.save_hf_mat == True:
            data["hf_mo_ene"] = self.mo_energy
            data["hf_mo_coeff"] = self.mo_coeff
            data["hf_dm"] = self.dm
            data["hf_ene"] = self.e_tot
        if self.save_loc_mat == True:
            data["loc_W"] = self.uo
            data["loc_L"] = self.o
            data["loc_fock"] = self.loc_fock
            pass
        if self.mbe_mode != 2:
            data["no_close_2b"] = len(self.twob_close_pair)
            data["no_remote_2b"] = len(self.twob_remote_pair)
            data["no_2b"] = data["no_close_2b"] + data["no_remote_2b"]
            data["no_3b"] = len(self.threeb_pair)
            total_twob = []
            for ipair in self.pairlist_offdiag:
                i = ipair//self.no
                j = ipair%self.no
                total_twob.append([i*self.no+i, i*self.no+j, j*self.no+j])
            data["total_twob"] = total_twob
            data["three_frag"] = self.threeb_pair
        print("Abs path of ML feature: %s"%os.path.abspath('ml_features.hdf5'))
        save_file(data, step)

# main function to compute MP2 integral TDNN features
def compute_energy_mbe(self, log):
    def info_ene():
        if self.mbe_mode != 2:
            log.info("    -----------------------------------------------")
            msg_list = [['MP2 correlation energy (Eh):', ''], ['Energy of 1-body:', '%.10f'%self.ene_1],
                            ['Energy of 2-body:', '%.10f'%self.ene_2], ['Energy of 3-body:', '%.10f'%self.ene_3]]
            if (self.twob_remote_pair != []):
                msg_list.extend([['-'*28, ''], ['Energy of long-range 2-body:', '%.10f'%self.ene_c2]])
            if self.mbe_mode == 0:
                msg_list.append(['Global correction:', '%.10f'%(self.ene_mp2-sum(self.mp2e_node[1:]))])
            msg_list.extend([['', '-'*len(list('%.10f'%self.ene_mp2))], ['Total MP2 correlation energy:', '%.10f'%self.ene_mp2]])
            print_align(msg_list, align='lr', indent=4, log=log)
            log.info("    -----------------------------------------------")
        time_list = [['energy computation', self.t_cal], 
                     ['data transmission', self.t_read+self.t_col],
                     ['residual iterations', self.t_res]]
        print_time(time_list, log)
    
    self.refer_molist = self.mo_list[:]
    self.refer_pairlist = self.pairlist[:]
    self.refer_pairlist_remote = self.pairlist_remote[:]
    self.refer_pairlist_close = self.pairlist_close[:]
    self.refer_pairlist_full = self.pairlist_full[:]

    if not (self.ml_test and self.mbe_mode == 2) or (self.ml_mp2int):
        #Implement OSV clus selections
        if self.mbe_mode == 0:
            mbe_mode = 'MBE(3)-OSV-MP2 with global correction'
        elif self.mbe_mode == 1:
            mbe_mode = 'g-MBE(3)-OSV-MP2 without global correction'
        elif self.mbe_mode == 2:
            mbe_mode = 'standard OSV-MP2'

        msg_list = ['\n-------------------------------OSV Many Body Expansion------------------------------',
                        '\nMBE mode: %s'%mbe_mode,
                        '\nBegin cluster selection...']
        log.info(''.join(msg_list))
        #Determination of OSV clusters
        clus_selection(self, log)

    if (self.ml_test) and (self.ml_mp2int == False):
        save_ml(self)
        sys.exit()
    else:
        get_buffers(self)
        msg = '\nBegin residual iterations...'
        msg += '\n    Begin MBE residual iterations...'
        log.info(msg)
        self.t_read = np.zeros(2)
        self.t_cal = np.zeros(2)
        self.t_col = np.zeros(2)
        t0 = time_now()
        if self.local_type == 0 or self.mbe_mode == 2:
            n3b_save = self.n3b_close
            self.n3b_close = 0
            count_3b_save = self.count_3b
            self.count_3b = [0]*self.no**2
        if self.mbe_mode == 0:
            self.etol_save = self.ene_tol
            self.ene_tol = 1e-6
        get_eg(self, self.oneblist_rank, 1, 'e', log) 
        if self.n2b_close > 0:
            get_eg(self, self.twoblist_close_rank, 20, 'e', log) 
        if self.mbe_mode != 2:
            if self.n3b_close > 0:
                get_eg(self, self.cluslist_3, 3, 'e', log)
            if self.n2b_remote > 0 and self.mbe_mode == 1:
                log.info('     ------------------------------Long range 2b correction------------------------------')
                if self.twob_remote_pair != []:
                    get_eg(self, self.twoblist_remote_rank, 21, 'e', log)
        Accumulate_GA_shm(self.win_ene, self.mp2e_node, self.mp2e_rank)
        comm.Barrier()
        if self.mbe_mode != 2:
            Acc_and_get_GA(self.mp2e_node)
        comm_shm.Barrier()
        print_time(["MBE residual", time_elapsed(t0)], log)

        if (self.mbe_mode != 1) and (self.local_type != 0):
            log.info('\n    Begin global residual correction...')
            if self.mbe_mode == 0:
                self.ene_tol = self.etol_save
            t1 = time_now()
            ene_close, ene_remote = get_mp2e_GA(self)
            self.mp2e_node[0] = ene_close+ene_remote
            self.mp2e_node[4] = ene_remote
            print_time(["global correction", time_elapsed(t1)], log)
        
        if self.local_type == 0 or self.mbe_mode == 2:
            self.n3b_close = n3b_save
            self.count_3b = count_3b_save
        self.ene_mp2 = self.mp2e_node[0]
        self.ene_1 = self.mp2e_node[1]
        self.ene_2 = self.mp2e_node[2]
        self.ene_3 = self.mp2e_node[3]
        self.ene_c2 = self.mp2e_node[4]
        self.t_res = time_elapsed(t0)
        info_ene()
        if self.ml_mp2int:
            save_ml(self)
    #print_time(["residual iterations", time_elapsed(t0)], log)
    
    return self.ene_mp2

def compute_gradient_mbe(self, get_d=True, get_r=True, log=None):
    cal_type = ''
    if get_d:
        cal_type += 'd'
    if get_r:
        cal_type += 'r'
    log.info('\nBegin OSV DM and dR with MBE...')
    self.t_read = np.zeros(2)
    self.t_cal = np.zeros(2)
    self.t_col = np.zeros(2)
    t0 = time_now()
    if (self.use_cposv) and get_r:
        #Compute S' and F'
        t1 = time_now()
        get_sf_cp_GA(self)
        print_time(["S' and F'", time_elapsed(t1)], log)
    #t1 = time_now()
    #print_time(["OSV DM and derivarive R", time_elapsed(t1)], log)
    get_eg(self, self.oneblist_rank, 1, cal_type, log)
    if self.n2b_close + self.n2b_remote > 0:
        get_eg(self, self.twoblist_rank, 2, cal_type, log)
    if self.n3b_close > 0:
        get_eg(self, self.cluslist_3, 3, cal_type, log)
    self.t_dr = time_elapsed(t0)
    if irank == 0: 
        time_list = [['DM and dR computation', self.t_cal], 
                     ['data transmission', self.t_read+self.t_col], 
                     ['OSV DM and dR', self.t_dr]]
        print_time(time_list, log)
        if self.direct_int and self.grad_cal and self.use_cposv:
            shutil.rmtree(self.dir_sf_cp)
    #t1 = time_now()

    #NON-MBE section
    self.mo_list = self.refer_molist
    self.pairlist = self.refer_pairlist
    self.pairlist_close = self.refer_pairlist_close
    self.pairlist_remote = self.refer_pairlist_remote
    log.info('\nStart computing dK and Yi...')
    t0 = time_now()
    self.gradient = get_gamma_GA(self)
    self.t_dk_yi = time_elapsed(t0)
    buffer_release(self, cal_type='gamma')

    t0 = time_now()
    self.gradient += mp2_dferi_GA(self)
    self.t_dferi_mp2 = time_elapsed(t0)

    self.gradient += ZCPL.AO_basis(self)
    return self.gradient
    



            
