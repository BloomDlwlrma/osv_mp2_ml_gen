# OSV-MP2 / OSV-CCSD 工作目录核心说明

本目录实现的是一条从 ORCA 结果回收、到 PySCF/OSV 计算、再到 HDF5 汇总与训练数据生产的完整量子化学工作流。它的核心目标不是重新发明电子结构理论，而是把 ORCA、MOKIT、PySCF 和 OSV 特征管线稳定地串起来，让 MP2/CCSD 相关能可以被批量提取、验证和重用。

---

## 1. `osvmp2` 的核心内容

### 1.1 它解决什么问题

`osvmp2` 是这个工作流的主线。它面向的是**局域电子相关**问题，核心思想是把常规 MP2 相关能拆到轨道对层面，再进一步用 **Orbital-Specific Virtuals, OSV** 压缩虚拟空间，使每个轨道对的相关贡献都能在较小的局部子空间中表达。

从代码结构上看，`osvmp2/` 目录里有几类典型组件：

- `opt_df.py`、`hf_ene.py`、`hf_grad.py`：负责 HF/DF 相关求解、能量和梯度。
- `OSVL.py`、`CPOSV.py`、`ZCPL.py`、`osvutil.py`：围绕 OSV 构造、局域化和工具函数展开。
- `grad_addons.py`、`ga_addons.py`、`mbe_addons.py`：补充梯度、几何优化和多体展开相关辅助逻辑。
- `loc/`、`berny/`、`geometric/`：几何优化与局域轨道相关支持。

### 1.2 物理与方法上的主线

`osvmp2` 的方法论可以概括为三层：

1. **HF 作为参考波函数**：先建立自洽场解，得到占据轨道和 Fock 结构。
2. **OSV 作为压缩表示**：对每个占据轨道建立轨道特异性虚拟空间，只保留最重要的虚拟方向，从而降低 MP2 相关计算的维度。
3. **按轨道对累积相关能**：相关能不是一次性以全局方式计算，而是以 pair 为单位组织，便于分块、并行和后续机器学习建模。

换句话说，`osvmp2` 的关键不是“算一个总 MP2 能量”，而是**把相关能结构化成可存储、可分片、可学习的 pair-level 数据**。

### 1.3 与代码实现对应的工作流

从 `work/batch_osvmp2_runner.py` 可以看出，实际运行方式是批处理式的：

- 每个分子会找到对应的 `xyz`、`hf_mat_*.chk`、`loc_var_*.chk`。
- 运行 `osvmp2/opt_df.py` 生成 `ml_features.hdf5` 和 `pair_energy.hdf5`。
- 计算日志里解析出 `RHF energy`、`MP2 correlation energy`、`Total energy`。
- 结果先写到每个 worker 的临时 slot，再移动到独立 staging 目录，最后由 collector 顺序合并到主 HDF5。

这套设计的重点是**避免并发删除文件导致的竞态条件**。也就是说，worker 负责快算，collector 负责稳合并。

**代表性实现片段（来自 `work/batch_osvmp2_runner.py`）：**
```python
def move_to_staging(src_path, dst_path):
    try:
        os.rename(src_path, dst_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src_path, dst_path)
        os.unlink(src_path)

def write_energy_record_hdf5(h5_path, mol_name, rhf_energy, mp2_corr_energy, total_energy, mode):
    with h5py.File(h5_path, 'a') as fout:
        if mol_name in fout:
            if mode == 'APPEND_MISSING':
                return 'skipped'
            del fout[mol_name]
        grp = fout.create_group(mol_name)
        grp.create_dataset('RHF_energy', data=rhf_energy)
        grp.create_dataset('MP2_correlation_energy', data=mp2_corr_energy)
        grp.create_dataset('Total_energy', data=total_energy)
        return 'written'

def worker(slot_id, task_queue, result_queue, base_env, allowed_ids=None):
    slot_dir = os.path.join(SLOT_BASE, f'slot_{slot_id}')
    os.makedirs(slot_dir, exist_ok=True)
    os.makedirs(STAGING_BASE, exist_ok=True)
    while True:
        try:
            mol_id = task_queue.get(timeout=3)
        except multiprocessing.queues.Empty:
            break
        if mol_id == 'STOP':
            break
        # prepare env and run opt_df.py in a per-slot temporary directory
        cmd = ['python', os.path.join(WORK_DIR, 'osvmp2', 'opt_df.py'), xyz_file]
        with open(run_log_path, 'w') as log_f:
            subprocess.run(cmd, env=env, cwd=slot_dir, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        # parse energies and move outputs to staging for collector to merge
        move_to_staging(tmp_out, staging_out)
```

上面片段展示了：使用 `os.rename()` 做同文件系统的原子移动；跨文件系统时回退到 `copy2`+`unlink`；以及把每个分子的输出先写入 slot 临时目录，再 `move_to_staging()` 到公共 `STAGING_BASE`，由 collector 安全合并。

---

### 1.4 输出是什么

`osvmp2` 路线最终会产出三类关键数据：

- `ml_features.hdf5`：机器学习用的局域特征。
- `pair_energy.hdf5`：轨道对相关能。
- `energy_record.hdf5`：每个分子的 RHF / MP2 / 总能量记录。

这些文件是后续训练、验证和误差分析的基础。

### 1.5 文献视角下的意义

如果结合 OSV-MP2 文献来理解，`osvmp2` 的核心价值在于：

- 用局域虚拟空间降低 MP2 的计算与存储成本。
- 把相关能表达成分子局域结构的函数，更适合大体系外推。
- 为后续基于 pair 的机器学习模型提供了物理一致的标签。

这也是为什么该目录不仅关心“算不算得出”，还特别关心“输出能不能稳定地被合并、校验和复用”。

---

## 2. `osvccsd` 的简短说明

`osvccsd` 是与 `osvmp2` 并行的一条更高精度路线，目标是从 ORCA 输出中提取 **CCSD**，以及可选的 **CCSD(T)** 三重激发修正信息，用作更高层级的参考能量。

从 `work/osvccsd/batch_osvccsd_runner.py` 可以看出，它的职责比较清晰：

- 批量读取 ORCA `.out` 文件。
- 调用 `01_extract_pair_corr_eng.py` 提取 pair correlation energy。
- 按 `osvccsd` 和 `osvccsdt` 两种模式分别生成：
  - `pair_energy_osvccsd.hdf5`
  - `pair_energy_osvccsdt.hdf5`
- 如果输入与已有主文件不一致，会先做缺失分子检查，再决定是否继续。

因此，`osvccsd` 可以理解为：**OSV-MP2 的高精度参考补充层**。它本身不是主线训练对象，但对于构建更可信的基准、分析 MP2 误差、以及扩展到更高精度目标很重要。


---

## 5. 快速理解流程

```text
ORCA 输出 / xyz
    ↓
MOKIT 转换 + PySCF 恢复
    ↓
osvmp2 计算局域相关特征与 pair energy
    ↓
HDF5 合并与校验
    ↓
机器学习训练 / 误差分析 / 高精度对照
```

如果你希望，我还可以继续把这份说明改成更像“文献综述风格”的版本，或者再补一版“按代码模块逐个解释”的版本。
