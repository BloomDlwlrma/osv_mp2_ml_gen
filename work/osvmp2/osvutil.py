import time
import types
import itertools
import psutil
import ctypes
import sys
import os
import warnings
import gc
import numpy as np
from numpy.linalg import svd, eigh, multi_dot
import pyscf
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.gto.moleintor import *
ddot = np.dot
import h5py
import mpi4py
mpi4py.rc.thread_level = 'single'
from mpi4py import MPI
if sys.version_info[0] >= 3:
    from functools import reduce
use_gpu = bool(int(os.environ.get("use_gpu", 0)))
if use_gpu:
    import cupy
    import cupyx
    import cupyx.scipy.linalg

#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore")
#Set up MPI environment
comm = MPI.COMM_WORLD
nrank = comm.Get_size()    # Size of communicator
irank = comm.Get_rank()    # Ranks in communicator
inode = MPI.Get_processor_name()     # Node where this MPI process runs
comm_shm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
irank_shm = comm_shm.rank # rank index in sub-comm
nrank_shm = comm_shm.size
nnode = nrank//nrank_shm


def set_omp(nthread):
    nthread = str(nthread)
    os.environ["OMP_NUM_THREADS"] = nthread
    os.environ["OPENBLAS_NUM_THREADS"] = nthread
    os.environ["MKL_NUM_THREADS"] = nthread
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthread
    os.environ["NUMEXPR_NUM_THREADS"] = nthread

def print_align(msg_list, align='l', align_1=None, indent=0, log=None, printout=True):
    if align_1 is None:
        align_1 = align
    align_list = []
    for align_i in [align_1, align]:
        align_format = []
        for i in list(align_i):
            if i == 'l':
                align_format.append('<')
            elif i == 'c':
                align_format.append('^')
            elif i == 'r':
                align_format.append('>')
        align_list.append(align_format)
    len_col = []
    for col_i in zip(*msg_list):
        len_col.append(max([len(str(i)) for i in col_i])+2)
    msg = ''
    for idx, msg_i in enumerate(msg_list):
        if idx == 0:
            align_i = align_list[0]
        else:
            align_i = align_list[1]
        msg += ' ' * indent
        msg += ''.join([('{:%s%d} '%(ali, li)).format(mi) for ali, li, mi in zip(align_i, len_col, msg_i)])
        if idx != len(msg_list)-1:
            msg += '\n'
    if printout:
        if log is None:
            print(msg)
        else:
            log.info(msg)
    return msg

def time_now():
    if sys.version_info < (3, 0):
        return np.asarray([time.clock(), time.time()])
    else:
        return np.asarray([time.process_time(), time.perf_counter()])
def time_elapsed(t0):
    return time_now() - t0

def print_time(time_list, log=None, left_align=False):
    def get_columns(name_i, time_i):
        return ['CPU time for %s'%name_i, '%.2f sec,'%time_i[0], 'wall time', '%.2f sec'%time_i[1]]
    if (len(time_list)==2) and (type(time_list[0])==str):
        time_list = [time_list]
    table_data = []
    for ti in time_list:
        name_i, time_i = ti
        table_data.append(get_columns(name_i, time_i))
    if left_align:
        indent = 0
    else:
        indent = 4
    print_align(table_data, align='lrcr', indent=indent, log=log)
    return time_now()

def print_size(var, varname, log=None):
    msg = 'Size of %s is %.2f MB'%(varname, var.size*8*1e-6)
    if log is None:
        print(msg)
    else:
        log.info(msg)

def mem_from_pid(pid):
    "Memory usgae with shared memory"
    mem_dic = psutil.Process(int(pid)).memory_info()
    return (mem_dic[0] - mem_dic[2] + mem_dic[2]/comm_shm.size)*1e-6

def print_test(a, varname=None):
    a = a.ravel()
    msg = '%.8E  %.8E  %.8E'%(min(a), max(a), np.mean(a))
    if varname is not None:
        msg = '%s: '%varname + msg
    print(msg)

def mem_node(pid_list=None):
    if pid_list is None:
        return psutil.virtual_memory()[3]*1e-6 #- mol.mem_init
    else:
        mem_total = 0
        for pid_i in pid_list:
            mem_total += mem_from_pid(pid_i)
        return mem_total
def print_mem(rank=None, pid_list=None, log=None, left_align=False):
    gc.collect()
    mem_used = mem_node(pid_list)
    msg = ', used memory: %.2f MB'%mem_used#lib.current_memory()[0]
    if rank is not None:
        msg = rank + msg
    if left_align == False:
        msg = '    ' + msg
    if log is None:
        print(msg)
    else:
        log.info(msg)
    
def get_mem_spare(mol, ratio=1):
    from osvmp2.ga_addons import get_shared, free_win
    gc.collect()
    #mem_avail = (mol.max_memory - psutil.virtual_memory()[3]*1e-6)/comm_shm.size
    #mem_avail = (mol.max_memory/comm_shm.size - mem_from_pid(os.getpid()))
    win_mem, mem_list = get_shared(comm_shm.size)
    mem_list[irank_shm] = mem_from_pid(os.getpid())
    comm_shm.Barrier()
    mem_total = sum(mem_list)
    comm_shm.Barrier()
    free_win(win_mem)
    mem_avail = (mol.max_memory - mem_total)/comm_shm.size
    return max(ratio*mem_avail, 1)

def get_buff_len(mol, size_sub, ratio, max_len, min_len=1, max_memory=None):
    if max_memory is None:
        max_memory = get_mem_spare(mol)
    if max_len is not None:
        return int(max(min(ratio*max_memory*1e6//(size_sub*8), max_len), min_len))

def generation_SuperMat(ijkl, matrix, blockdim, ndim):  
    i, j, k, l = ijkl
    SuperMat = np.empty((blockdim[i]+blockdim[j], blockdim[k]+blockdim[l]))
    SuperMat[:blockdim[i], :blockdim[k]] = matrix[i*ndim+k]
    SuperMat[blockdim[i]:, :blockdim[k]] = matrix[j*ndim+k] 
    SuperMat[:blockdim[i], blockdim[k]:] = matrix[i*ndim+l]
    SuperMat[blockdim[i]:, blockdim[k]:] = matrix[j*ndim+l]
    return SuperMat

def contigous_trans(a, order=None):
    if order == None:
        b = a.T
    else:
        b = a.transpose(order)
    a = a.reshape(b.shape)
    a[:] = b
    return a


def half_trans(mol, feri, mo_coeff, lmo_close, fit_close, slice_i, i, buf_feri=None, buf_moco=None, dot=np.dot, out=None):
    def get_ao_domains(be0, be1, lmo_close, i):
        lmo_slice = lmo_close[i]
        if (lmo_slice is None):
            return None
        elif (lmo_slice[0][0] >= be1) or (lmo_slice[-1][-1] <= be0):
            return None
        else:
            cal_slice = []
            for BE0, BE1 in lmo_slice:
                    if BE0 >= be1: break
                    if BE1 > be0:
                        cal_slice.append([max(be0, BE0), min(be1, BE1)])
            #if irank == 0: print(be0, be1, lmo_slice, cal_slice)
            return cal_slice
    al0, al1, be0, be1 = slice_i
    be_idx = [None] * (be1+1)
    for idx, be in enumerate(range(be0, be1+1)):
        be_idx[be] = idx
    cal_slice = get_ao_domains(be0, be1, lmo_close, i)
    if cal_slice is None or cal_slice == []:
        return None
    else:
        nao0 = al1 - al0
        nao1 = sum([(BE1-BE0) for BE0, BE1 in cal_slice])
        #naux = sum([(p1-p0) for p0, p1 in fit_close[i]])
        naux = feri.shape[-1]
        if buf_feri == None:
            feri_tmp = np.empty((nao1, nao0, naux))
        else:
            feri_tmp = buf_feri[:nao0*nao1*naux].reshape(nao1, nao0, naux)
        if buf_moco == None:
            moco_tmp = np.empty(nao1)
        else:
            moco_tmp = buf_moco[:nao1]
        
        idx_BE0 = 0
        for BE0, BE1 in cal_slice:
            idx_BE1 = idx_BE0 + (BE1-BE0)
            moco_tmp[idx_BE0:idx_BE1] = mo_coeff[BE0:BE1, i]
            idx_be0, idx_be1 = be_idx[BE0], be_idx[BE1]
            feri_tmp[idx_BE0:idx_BE1] = feri[idx_be0:idx_be1]
            '''idx_p0 = 0
            for p0, p1 in fit_close[i]:
                    idx_p1 = idx_p0 + (p1-p0)
                    feri_tmp[idx_BE0:idx_BE1, :, idx_p0:idx_p1] = feri[idx_be0:idx_be1, :, p0:p1]
                    idx_p0 = idx_p1'''
            idx_BE0 = idx_BE1

        try:
            if out is None:
                    return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1)).reshape(nao0, naux)
            else:
                    return dot(moco_tmp.T, feri_tmp.reshape(nao1, -1), out=out).reshape(nao0, naux)
        except ValueError:
            print('DAMIT', cal_slice)

def get_auxshell_slice(auxmol):
    aux_loc = make_loc(auxmol._bas, 'sph')
    shell_seg = []
    naux_seg = []
    for s0 in range(auxmol.nbas):
        s1 = s0 + 1
        shell_seg.append([s0, s1])
        naux_seg.append(aux_loc[s1]-aux_loc[s0])
    if len(naux_seg) < nrank:
        shell_slice = get_slice(rank_list=range(nrank), job_list=shell_seg)
        for idx, s_i in enumerate(shell_slice):
            if s_i is not None:
                shell_slice[idx] = sorted(list(set(reduce(lambda x, y :x+y, s_i))))
    else:
        shell_slice = OptPartition(nrank, shell_seg, naux_seg)[0]
        if len(shell_slice) < nrank:
            shell_slice = get_slice(rank_list=range(nrank), job_list=shell_slice)
            for rank_i, s_i in enumerate(shell_slice):
                if s_i is not None:
                    shell_slice[rank_i] = s_i[0]
    aux_slice = [None]*nrank
    aux_address = [None]*auxmol.nao_nr()
    for rank_i, shell_i in enumerate(shell_slice):
        if shell_i is not None:
            s0, s1 = shell_i[0], shell_i[-1]
            aux0, aux1 = aux_loc[s0], aux_loc[s1]
            aux_slice[rank_i] = []
            for idx, p in enumerate(range(aux0, aux1)):
                aux_slice[rank_i].append(p)
                aux_address[p] = [rank_i, [idx, idx+1]]
    return aux_slice, aux_address, shell_slice

def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError as e:
        pass

def flip_ij(i,j,mat,blockdim):
    if(i!=j):
        result=np.empty((blockdim[i]+blockdim[j],blockdim[i]+blockdim[j]))
        result[blockdim[j]:,blockdim[j]:] = mat[:blockdim[i],:blockdim[i]].T
        result[:blockdim[j],:blockdim[j]] = mat[blockdim[i]:,blockdim[i]:].T
        result[:blockdim[j],blockdim[j]:] = mat[:blockdim[i],blockdim[i]:].T
        result[blockdim[j]:,:blockdim[j]] = mat[blockdim[i]:,:blockdim[i]].T
    else:
        result = mat
    return result

def normalize_mo_orca(mol, mo_coeff):
    def fact2(k):
        """
        Compute double factorial: k!! = 1*3*5*....k
        """
        from operator import mul
        return reduce(mul, range(k, 0, -2), 1.0)

    def atom_list_converter(self):
        """
        in ORCA 's', 'p' orbitals don't require normalization.
        'g' orbitals need to be additionally scaled up by a factor of sqrt(3).
        https://orcaforum.cec.mpg.de/viewtopic.php?f=8&t=1484
        """
        ao_loc = make_loc(mol._bas, 'sph')
        for ib in range(mol.nbas):
            al0, al1 = ao_loc[ib], ao_loc[ib+1]
            l = mol.bas_angular(ib)
            
            for primitive in shell['DATA']:
                primitive[1] /= sqrt(fact2(2*l-1))
                if l == 4:
                    primitive[1] *= sqrt(3)

    def f_normalize(self, mo_coeff):
        """
        ORCA use slightly different sign conventions:
            F(+3)_ORCA = - F(+3)_MOLDEN
            F(-3)_ORCA = - F(-3)_MOLDEN
        """
        mo_coeff[5] *= -1
        mo_coeff[6] *= -1
        return super(Orca, self).f_normalize(mo_coeff)

    def g_normalize(self, mo_coeff):
        """
        ORCA use slightly different sign conventions:
            G(+3)_ORCA = - G(+3)_MOLDEN
            G(-3)_ORCA = - G(-3)_MOLDEN
            G(+4)_ORCA = - G(+4)_MOLDEN
            G(-4)_ORCA = - G(-4)_MOLDEN
        """
        mo_coeff[5] *= -1
        mo_coeff[6] *= -1
        mo_coeff[7] *= -1
        mo_coeff[8] *= -1
        return super(Orca, self).g_normalize(mo_coeff)

def get_coords_from_mol(mol, coord_only=False, info=""):
    atm_list = []
    for atm in range(mol.natm):
        atm_list.append(mol.atom_pure_symbol(atm))
    xyz_list = mol.atom_coords()*lib.param.BOHR
    coord_list = [[atm_list[atm], "%.9f"%x, "%.9f"%y, "%.9f"%z] \
                  for atm, (x, y, z) in enumerate(xyz_list)]
    coord_str = print_align(coord_list, align='lrrr', printout=False) 
    if coord_only:
        return coord_str
    else:
        coords_msg = "%d\n%s\n"%(len(atm_list), info)
        coords_msg += coord_str + "\n"
        return coords_msg

def get_ovlp(mol):
    '''Overlap matrix
    '''
    return mol.intor_symmetric('int1e_ovlp')

def str_letter(msg):
    return (''.join(x for x in msg if x.isalpha())).lower()


def list2seg(list_i):
    if len(list_i) == 1:
        idx = list(list_i)[0]
        return [[idx, idx+1]]
    seg_list = []
    seg_i = []
    i_pre = -2
    for idx, i in enumerate(list_i):
        if i != i_pre+1 and idx != 0:
            seg_list.append([seg_i[0], seg_i[-1]+1])
            seg_i = [i]
        else:
            seg_i.append(i)
        if idx == len(list_i)-1:
            seg_list.append([seg_i[0], seg_i[-1]+1])
        i_pre = i
    return seg_list


def get_slice(rank_list, job_size=None, job_list=None):
    if type(job_list) == type(None):
        job_list = range(job_size)
    else:
        job_size = len(job_list)

    n_rank = len(rank_list)
    if n_rank > nrank_shm:
        rank_list = []
        for shm_rank_i, node_i in itertools.product(range(nrank//nnode), range(nnode)):
            rank_list.append(node_i*nrank_shm + shm_rank_i)
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

def get_omul_seg(self, slice_i, stype):
    #if stype == 'aux_mp2':
    if 'aux' in stype:
        omul_list = self.omul_aux
        dir_omul = self.dir_omul_aux
    else:
        omul_list = self.omul_mo
        dir_omul = self.dir_omul_mo
    omul_list = np.asarray(omul_list)
    rank_all = omul_list[slice_i]
    rank_list = [rank_all[0]]#list(set(rank_all.tolist()))
    seg_list = []
    seg_i = []
    for idx, rank_i in enumerate(rank_all):
        seg_i.append(slice_i[idx])
        if rank_i == rank_all[-1]:
            #rank_list.append(rank_i)
            seg_list.append(slice_i[idx:])
            break
        elif rank_all[idx+1] != rank_i:
            rank_list.append(rank_all[idx+1])
            seg_list.append(seg_i)
            seg_i = []
    file_list = ['%s/omul_%s_%d.tmp'%(dir_omul, stype, rank_i) for rank_i in rank_list]
    return file_list, seg_list

def getints3c_test(intor_name, atm, bas, env, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)
        if 'ssc' in intor_name or 'spinor' in intor_name:
            bas = numpy.asarray(numpy.vstack((bas,bas)), dtype=numpy.int32)
            shls_slice = (0, nbas, 0, nbas, nbas, nbas*2)
            nbas = bas.shape[0]
    else:
        assert(shls_slice[1] <= nbas and
               shls_slice[3] <= nbas and
               shls_slice[5] <= nbas)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
        if 'ssc' in intor_name:
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
        elif 'spinor' in intor_name:
            # The auxbasis for electron-2 is in real spherical representation
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'sph')

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        shape = (naoi, naoj, naok, comp)
    else:
        aosym = 's2ij'
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok, comp)
    order = 'F'
    #order = 'C'
    if 'spinor' in intor_name:
        mat = numpy.ndarray(shape, numpy.complex, out, order=order)
        drv = libcgto.GTOr3c_drv
        fill = getattr(libcgto, 'GTOr3c_fill_'+aosym)
    else:
        mat = numpy.ndarray(shape, numpy.double, out, order=order)
        drv = libcgto.GTOnr3c_drv
        fill = getattr(libcgto, 'GTOnr3c_fill_'+aosym)

    if mat.size > 0:
        # Generating opt for all indices leads to large overhead and poor OMP
        # speedup for solvent model and COSX functions. In these methods,
        # the third index of the three center integrals corresponds to a
        # large number of grids. Initializing the opt for the third index is
        # not necessary.
        if cintopt is None:
            if '3c2e' in intor_name:
                # int3c2e opt without the 3rd index.
                cintopt = make_cintopt(atm, bas[:max(i1, j1)], env, intor_name)
            else:
                cintopt = make_cintopt(atm, bas, env, intor_name)

        drv(getattr(libcgto, intor_name), fill,
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*(shls_slice[:6])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    
    if comp == 1:
        mat = mat[:,:,:,0]
    else:
        mat = numpy.rollaxis(mat, -1, 0)
    return mat

def aux_e2(mol=None, auxmol=None, intor='int3c2e', aosym='s1', comp=None, 
              cintopt=None, shls_slice=None, hermi=0, out=None):
    
    ao_loc = None
    if '3c' in intor:
        intor = mol._add_suffix(intor)
        if shls_slice == None:
            shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                                auxmol._atm, auxmol._bas, auxmol._env)
        return getints3c(intor, atm, bas, env, shls_slice, comp, aosym, ao_loc, cintopt, out)
    elif '2c' in intor:
        if mol is None:
            mol = auxmol
        intor = mol._add_suffix(intor)
        return getints2c(intor, mol._atm, mol._bas, mol._env, shls_slice, comp, hermi, ao_loc, cintopt, out)

def get_max_sum(array, n, K):
    def check(mid, array, n, K): 
        count = 0
        sum = 0
        # If individual element is greater maximum possible sum 
        for i in range(n): 
            if (array[i] > mid): 
                return False
            # Increase sum of current sub - array 
            sum += array[i] 
            # If the sum is greater than mid increase count 
            if (sum > mid): 
                count += 1
                sum = array[i] 
        count += 1
        if (count <= K): 
            return True
        return False
    start = 1
    end = 0
    #Initialise end
    for i in range(n): 
            end += array[i] 
    # Answer stores possible maximum sub array sum 
    answer = 0
    while (start <= end): 
        mid = (start + end) // 2
        # If mid is possible solution, put answer = mid; 
        if (check(mid, array, n, K)): 
            answer = mid 
            end = mid - 1
        else: 
            start = mid + 1
    return answer 

def OptPartition(n_rank, shell_seg, ao_pair_seg, match_num=True):
    def collect_slice(shell_seg, ao_pair_seg, max_sum):
        shell_i = []
        len_si = 0
        shell_slice = []
        len_list = []
        for idx, len_i in enumerate(ao_pair_seg):
            shell_i.extend(shell_seg[idx])
            len_si += len_i
            idx_next = idx+1
            if idx_next > len(ao_pair_seg)-1:
                idx_next = idx
            if (len_si + ao_pair_seg[idx_next] > max_sum) or idx==(len(ao_pair_seg)-1):
                shell_slice.append(sorted(list(set(shell_i))))
                shell_i = []
                len_list.append(len_si)
                len_si = 0
        return shell_slice, len_list
    
    #max_sum = get_max_sum(ao_pair_seg, len(ao_pair_seg), n_rank)
    max_sum = sum(ao_pair_seg)//n_rank + 10
    shell_slice, len_list = collect_slice(shell_seg, ao_pair_seg, max_sum)
    len_slice = len(len_list)
    shell_pre = shell_slice
    len_pre = len_list
    step = 0
    var_i = 10
    if (len_slice != n_rank): #and match_num:
        while len_slice != n_rank:
            shell_slice, len_list = collect_slice(shell_seg, ao_pair_seg, max_sum)
            len_slice = len(len_list)
            if len_slice < n_rank:
                    if (step > 20) and (len(len_pre)>n_rank):
                        break
                    max_sum -= var_i
                    shell_pre = shell_slice
                    len_pre = len_list
            elif len_slice > n_rank:
                    max_sum += var_i
                    shell_pre = shell_slice
                    len_pre = len_list
            step += 1
            var_i = var_i//2
            if var_i == 0:
                    var_i = 1
    return shell_slice, len_list

def check_read(fil_name, dat_name):
    with h5py.File(fil_name, 'r') as f:
        if dat_name in f.keys():
            return True
        else:
            return False

def shell_chk(fil_name='sparse.chk', dat_name='shell_slice_hf', mol=None, shell_slice=None, op='w'):
    if op == 'r':
        from osvmp2.int_prescreen import alloc_shell
        with h5py.File(fil_name, op) as f:
            shell_flat = np.asarray(f[dat_name])
        ao_loc = make_loc(mol._bas, 'sph')
        naop_shell = []
        for a0, a1, b0, b1 in shell_flat:
            naop_shell.append((ao_loc[a1]-ao_loc[a0])*(ao_loc[b1]-ao_loc[b0]))
        return alloc_shell(shell_flat, naop_shell)
    else:
        shell_flat = []
        for si in shell_slice:
            if si is not None:
                    shell_flat.extend(si)
        shell_flat = np.asarray(shell_flat)
        with h5py.File(fil_name, op) as f:
            f.create_dataset(dat_name, shape=(len(shell_flat), 4), dtype='i')
            f[dat_name].write_direct(shell_flat)

    
def mem_control(mol, nocc, naoaux, slice_rank, rank, max_memory=None, nbfit=None):
    def get_shell_seg(slice_rank):
        shell_idx = [None]*(mol.nbas+1)
        shell_seg = []
        for a0, a1, b0, b1 in slice_rank:
            idx = shell_idx[a0]
            if idx is None:
                shell_idx[a0] = len(shell_seg)
                shell_seg.append([a0, a1, [[b0, b1]]])
            else:
                shell_seg[idx][-1].append([b0, b1])
        return shell_seg
    def mem_con_ashell(slice_rank, size_feri=None, max_nao=None, nao0_list=None):
        shell_seg = get_shell_seg(slice_rank)
        shellslice_a = []
        for a0, a1, b_seg in shell_seg:
            max_nao1 = max([(ao_loc[b1]-ao_loc[b0]) for b0, b1 in b_seg])
            al0, al1 = ao_loc[a0], ao_loc[a1]
            nao0 = al1 - al0
            slice_a = False
            if max_nao is None:
                if nao0*max_nao1*naoaux > size_feri:
                    slice_a = True
            else:
                if max_nao < max(nao0_list):
                    slice_a = True
            if slice_a:
                nao_i = 0
                ai_init = a0
                shell_temp = []
                for ai in range(a0, a1):
                    add_slice = False
                    if max_nao is None:
                        if (ao_loc[ai]-ao_loc[ai_init])*max_nao1*naoaux > size_feri:
                            add_slice = True
                    else:
                        if (nao_i + ao_loc[ai+1] - ao_loc[ai]) > max_nao:
                            add_slice = True
                    if add_slice:
                        for b0, b1 in b_seg:
                            shell_temp.append([ai_init, ai, b0, b1])
                        ai_init = ai
                        nao_i = ao_loc[ai+1] - ao_loc[ai]
                    else:
                        nao_i += ao_loc[ai+1] - ao_loc[ai]
                    if ai == a1-1:
                        for b0, b1 in b_seg:
                            shell_temp.append([ai_init, ai+1, b0, b1])
                shellslice_a.extend(shell_temp)
            else:
                for b0, b1 in b_seg:
                    shellslice_a.append([a0, a1, b0, b1])
        return shellslice_a
    
    def mem_con_bshell(shellslice_a, size_feri):
        shellslice_rank = []
        for shell_i in shellslice_a:
            a0, a1, b0, b1 = shell_i
            al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
            nao0 = al1 - al0
            nao1 = be1 - be0
            if nao0*nao1*naoaux > size_feri:
                bi_0 = b0
                shell_temp = []
                for bi in range(b0, b1):
                    if nao0*(ao_loc[bi]-ao_loc[bi_0])*naoaux > size_feri:
                        shell_temp.append([a0, a1, bi_0, bi])
                        bi_0 = bi
                    if bi == b1-1:
                        shell_temp.append([a0, a1, bi_0, bi+1])
                shellslice_rank.extend(shell_temp)
            else:
                shellslice_rank.append([a0, a1, b0, b1])
        return shellslice_rank
    
    def get_size_feri(shellslice_rank):
        naop_list = []
        for shell_i in shellslice_rank:
            al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
            naop_list.append((al1-al0)*(be1-be0))
        return max(naop_list)*naoaux
    def get_size_ialp(shellslice_rank, max_nao, prod_no_naux, nbfit=None):
        shell_check = [False]*mol.nbas
        nal_list = []
        for shell_i in shellslice_rank:
            if shell_check[shell_i[0]] is False:
                al0, al1, be0, be1 = [ao_loc[s] for s in shell_i]
                nal_list.append(al1-al0)
        if max_nao >= sum(nal_list):
            max_nao = sum(nal_list)
        elif max_nao <= max(nal_list):
            max_nao = max(nal_list)
        else:
            max_nal_pre = max(nal_list)
            for n in range(2, len(nal_list)):
                sum_list = []
                for idx0 in range(len(nal_list)-n+1):
                    sum_list.append(sum(nal_list[idx0:idx0+n]))
                max_nal = max(sum_list)
                if max_nao < max_nal:
                    max_nao = max_nal_pre
                    break
                max_nal_pre = max_nal

        size_ialp = max_nao*prod_no_naux
        if nbfit is None:
            size_ialp_f = max(nal_list)*prod_no_naux
        else:
            size_ialp_f = max(nal_list)*max([nfit_i for nfit_i in nbfit if nfit_i is not None])
        return size_ialp, size_ialp_f
    
    ao_loc = make_loc(mol._bas, 'sph')
    if max_memory is None:
        max_memory = 0.9*get_mem_spare(mol)
    if type(rank) == float:
        r_feri = rank
        size_feri = r_feri*max_memory*1e6/8
        shellslice_a = mem_con_ashell(slice_rank, size_feri=size_feri)
        shellslice_rank = mem_con_bshell(shellslice_a, size_feri)
        return get_size_feri(shellslice_rank), shellslice_rank
    elif rank == "half_trans":
        #r_feri = 0.2
        r_ialp = 0.6
        mem_cost_pre = 0
        for istep in range(int((1-r_ialp)//0.05) + 1):
            nao0_list = []
            naop_list = []
            nao_rank = 0
            al0_pre = -1
            for a0, a1, b0, b1 in slice_rank:
                al0, al1, be0, be1 = [ao_loc[s] for s in [a0, a1, b0, b1]]
                nao0 = al1 - al0
                nao1 = be1 - be0
                nao0_list.append(nao0)
                naop_list.append(nao0*nao1)
                if al0 != al0_pre:
                    nao_rank += nao0
                al0_pre= al0
            #size_feri = max(naop_list)*naoaux
            if nbfit is None:
                prod_no_naux = nocc*naoaux
                size_ialp_f = max(nao0_list)*prod_no_naux
            else:
                nfit_list = [nfit_i for nfit_i in nbfit if nfit_i is not None]
                prod_no_naux = sum(nfit_list)
                size_ialp_f = max(nao0_list)*max(nfit_list)
            size_ialp = max(nao0_list)*prod_no_naux
            
            if (size_ialp + size_ialp_f)*8*1e-6 < r_ialp*max_memory:
                max_nao = int((r_ialp*max_memory - size_ialp_f*8*1e-6)//(prod_no_naux*8*1e-6))
                max_nao = min(max_nao, nao_rank)
            else:
                nal_list = []
                for a0, a1, b0, b1 in slice_rank:
                    for ai in range(a0, a1):
                        nal_list.append(ao_loc[ai+1] - ao_loc[ai])
                size_sub_f = size_ialp_f//max(nao0_list)
                max_nao = int(max(r_ialp*max_memory//((size_sub_f+prod_no_naux)*8*1e-6), max(nal_list)))
            if max_nao < max(nao0_list):
                shellslice_a = mem_con_ashell(slice_rank, max_nao=max_nao, nao0_list=nao0_list)
            else:
                shellslice_a = slice_rank
            size_ialp, size_ialp_f = get_size_ialp(shellslice_a, max_nao, prod_no_naux, nbfit)
            size_feri = (max_memory - ((size_ialp+size_ialp_f)*8*1e-6))//(8*1e-6)
            mem_short = False
            if size_feri < 0:
                mem_short = True
            shellslice_rank = mem_con_bshell(shellslice_a, max(size_feri/2, 1))
            size_feri = get_size_feri(shellslice_rank)
            mem_cost = (size_ialp+size_ialp_f+2*size_feri)*8*1e-6
            if mem_short:
                raise MemoryError("[Rank %d, PID %d] Not enough memory for half-trans (current: %.2f MB; required: %.2f MB)"%(irank, os.getpid(), max_memory, mem_cost))
            #if irank == 0: print(r_ialp, "%.2f MB, %.2f MB"%((size_ialp+size_ialp_f+2*size_feri)*8*1e-6, max_memory))
            if (mem_cost > max_memory) or (r_ialp == 1) or (mem_cost == mem_cost_pre):
                break
            else:
                r_ialp += 0.05
                mem_cost_pre = mem_cost
                
        return size_ialp, size_feri, shellslice_rank
    elif rank == "derivative_feri":
        r_ialp = 0.3
        mem_cost_pre = 0
        for istep in range(int((1-r_ialp)//0.05) + 1):
            nao0_list = []
            naop_list = []
            nao_rank = 0
            al0_pre = -1
            for a0, a1, b0, b1 in slice_rank:
                al0, al1, be0, be1 = [ao_loc[s] for s in [a0, a1, b0, b1]]
                nao0 = al1 - al0
                nao1 = be1 - be0
                nao0_list.append(nao0)
                naop_list.append(nao0*nao1)
                if al0 != al0_pre:
                    nao_rank += nao0
                al0_pre= al0
            #size_feri = max(naop_list)*naoaux
            size_ialp = max(nao0_list)*nocc*naoaux
            if size_ialp*8*1e-6 < r_ialp*max_memory:
                max_nao = max(nao0_list)
            else:
                nal_list = []
                for a0, a1, b0, b1 in slice_rank:
                    for ai in range(a0, a1):
                        nal_list.append(ao_loc[ai+1] - ao_loc[ai])
                max_nao = int(max(r_ialp*max_memory//(nocc*naoaux*8*1e-6), max(nal_list)))
                size_ialp = max_nao*nocc*naoaux
            if max_nao < max(nao0_list):
                shellslice_a = mem_con_ashell(slice_rank, max_nao=max_nao, nao0_list=nao0_list)
            else:
                shellslice_a = slice_rank
            size_ialp = get_size_ialp(shellslice_a, max_nao, nocc*naoaux)[1]
            size_feri = (max_memory - (size_ialp*8*1e-6))//(4*8*1e-6)
            shellslice_rank = mem_con_bshell(shellslice_a, size_feri)
            size_feri = 4*get_size_feri(shellslice_rank)
            mem_cost = (size_ialp+size_feri)*8*1e-6
            #if irank == 0: print(r_ialp, "%.2f MB, %.2f MB"%((size_ialp+size_feri)*8*1e-6, max_memory))
            if (mem_cost > max_memory) or (r_ialp == 1) or (mem_cost == mem_cost_pre):
                break
            else:
                r_ialp += 0.05
                mem_cost_pre = mem_cost
        return size_ialp, size_feri, shellslice_rank
    else:
        raise IOError('No memory control scheme for process: "%s"'%rank)

def slice2seg(mol, shell_slice, max_nao=None):
    shell_check = [False]*mol.nbas
    shell_seg = []
    seg_i = []
    for idx, (a0, a1, b0, b1) in enumerate(shell_slice):
        if shell_check[a0]:
            seg_i[-1].append([b0, b1])
        else:
            if idx != 0:
                shell_seg.append(seg_i)
            seg_i = [a0, a1, [[b0, b1]]]
            shell_check[a0] = True
        if idx == (len(shell_slice)-1):
            shell_seg.append(seg_i)
    if max_nao is None:
        return shell_seg
    else:
        ao_loc = make_loc(mol._bas, 'sph')
        SHELL_SEG = []
        SEG_i = []
        nao_i = 0
        for idx, seg_i in enumerate(shell_seg):
            al0, al1 = [ao_loc[s] for s in seg_i[:2]]
            if (nao_i + al1 - al0) > max_nao:
                SHELL_SEG.append(SEG_i)
                SEG_i = [seg_i]
                nao_i = al1 - al0
            else:
                SEG_i.append(seg_i)
                nao_i += al1 - al0
            if idx == (len(shell_seg)-1):
                SHELL_SEG.append(SEG_i)
        return SHELL_SEG
        
                
            
            
        
