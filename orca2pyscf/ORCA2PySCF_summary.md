# ORCA → PySCF 转换工具 概要

本目录实现了将 ORCA 生成的轨道/矩阵（MKL/GBW 转换为 MOKIT 可读格式）导入到 PySCF，并保存为统一的 HDF5 chk 文件，供后续的 OSV-MP2 / ML 特征生成管线使用。

---

**关键脚本与文件**

- `orca2pyscf.py` - 轻量示例驱动脚本：读取 ORCA 的 MKL 文件，使用 `mkl2fch` 将 MKL 转为 `.fch`，用 `fch2py` 读取 MO/LO，并将 HF/LO 数据保存为 HDF5 chk 文件。
- `workflow_orca2pyscf.py` - 生产级工作流脚本：包含更健壮的环境变量参数、辅助基映射（`DEFAULT_AUXBASIS`）、重建容差与策略（`ORCA_RECON_TOL`、`ORCA_RECON_POLICY`），并写入 `meta` 元数据到 chk 文件。
- `validate_ao_lo_consistency.py` - 校验工具：批量验证 HF chk 与 LO chk 的 AO/LO 一致性，计算一系列误差指标并生成 JSON 报告（`ao_lo_error_metrics_notes.tex` 中定义的指标）。
- `ao_lo_error_metrics_notes.tex` - 论文/报告级说明：列出并定义了所有 AO/LO 误差指标（如 `ao_orth_err`、`recon_err`、`uo_cond`、`subspace_proj_err` 等）。
- `mybasis_verify.json` - 支持的基组列表（目前包含 `cc-pVTZ` 与 `def2-TZVP`）。
- `mkl2chk.cmd`、`init_mokit.sh` 等脚本/日志目录：用于环境初始化与批处理记录。

---

## 功能要点（核心流程）

1. 读取输入：ORCA 输出文件（通常经过 `orca_2mkl` 生成的 `.mkl`），或直接读取单点 `xyz` 文件以构建 PySCF `Mole`。
2. 基组与辅助基设置：使用 `DEFAULT_AUXBASIS` 将常用 AO 基组映射到对应的 JK-fit / RI 辅助基。脚本中通过 `get_aux_orca()` 自动对每种原子分配辅助基集。
3. 转换流程：
   - 使用 MOKIT 提供的 `mkl2fch` 将 `.mkl` 转为 `.fch`。
   - 用 `fch2py` 读取 `.fch`，得到 `mo_coeff`、局域轨道（`o`/`uo`）等。
   - 用 PySCF 构建 `mf` 对象，恢复 `dm`、`fock`、`hcore`、`escf` 等字段（如 `mf.dm = mf.make_rdm1()`）。
4. 本地轨道（LO）处理：
   - 将读取的 LO 投影到占据子空间，计算 `uo = C_occ^T S O` 并检查重构误差（`recon_err`）。
   - 计算局域 Fock 矩阵并提取 LO 能量 `eo`。
   - 支持两种重建策略：`STRICT`（失败则抛错）与 `SKIP`（打印警告并继续）。重建容差由 `ORCA_RECON_TOL` 控制。
5. 保存格式：生成两个 HDF5 chk 文件（HF：`hf_mat.chk`，LO：`loc_var.chk`），关键数据组织如下：
   - `hf_mat.chk` -> `scf/` 下包含 `dm`, `mo_energy`, `mo_coeff`, `mo_occ`, `mocc`, `e_tot`，并在 `meta` 中写入 `chk_kind`, `basis_key`, `nao`, `no`, `natom`, `atom_signature`, `auxbasis_hf`, `auxbasis_mp2`, `source_mkl` 等属性。
   - `loc_var.chk` -> 包含 `uo`, `o`, `loc_fock`, `eo`，并写入相应 `meta` 属性（包括源 MKL 路径）。

   **代表性实现片段（来自脚本）：**
   ```python
   def mkl2fch(mklfile, path_mkl2fch):
      if not Path(mklfile).is_file():
         raise ValueError("MKL file not found")
      subprocess.run([path_mkl2fch, mklfile], check=True)

   def load_orca_mo(mf, mklfile, path_mkl2fch):
      mkl2fch(mklfile, path_mkl2fch)
      fchfile = Path(mklfile).with_suffix('.fch')
      mf.mo_coeff = fch2py(fchfile, mf.mol.nao, mf.mol.nao, 'a')
      ovlp = mf.mol.intor_symmetric('int1e_ovlp')
      check_orthonormal(mf.mol.nao, mf.mol.nao, mf.mo_coeff, ovlp)
      mf.dm = mf.make_rdm1()
      mf.fock = mf.get_fock()
      mf.escf = 0.5 * np.sum(mf.dm * (mf.get_hcore() + mf.fock)) + mf.mol.energy_nuc()

   def save_chkhf(mf, filename='hf_mat.chk'):
      hfe = np.array([mf.escf])
      access_chkfile(filename, 'w', [mf.dm, mf.mo_energy, mf.mo_coeff, mf.mo_occ, mf.mocc, hfe])
      write_chk_metadata(filename, mf, chk_kind='hf', source_mkl=getattr(mf, '_source_mkl', None))
   ```
6. 验证工具：`validate_ao_lo_consistency.py` 对批量分子做严格的 AO/LO 一致性校验，计算并输出一组指标到 JSON（可选择只保存失败条目）。

---

## 重要参数与环境变量（运行示例）

必须提供的环境变量（`workflow_orca2pyscf.py` 在 `__main__` 中要求）：
- `WORK_ROOT`、`OUTPUT_ROOT`、`XYZ_ROOT`、`MOKIT_BIN`
- `METHOD`（如 `ccsdt`）、`BASIS_TAG`（如 `631gss`）、`AO_BASIS`
- `START_MOL`、`END_MOL`

与 LO 重建控制：
- `ORCA_RECON_TOL`（默认 `1e-6`）
- `ORCA_RECON_POLICY`（`STRICT` 或 `SKIP`，默认 `STRICT`）

示例运行命令（在合适的 shell 环境下导出环境变量后执行）：

```bash
export WORK_ROOT=/path/to/work
export OUTPUT_ROOT=/path/to/output
export XYZ_ROOT=/path/to/xyz
export MOKIT_BIN=/path/to/mokit/bin/mkl2fch
export METHOD=mp2int
export BASIS_TAG=ccpvtz
export AO_BASIS=cc-pVTZ
export START_MOL=1
export END_MOL=100
python3 workflow_orca2pyscf.py
```

或单分子调试：

```bash
python3 orca2pyscf.py
# 示例脚本内含 driver 主程序（默认示例文件名）
```

---

## 输出与下游消费

- 生成的 `hf_mat.chk` 与 `loc_var.chk` 被后续 `OSV-MP2` 特征生成代码复用（例如 `T-dNN` 项目中 `ml_feature.py` 期待特定的 HDF5 组名与元数据）。
- `validate_ao_lo_consistency.py` 生成的 JSON 报告可用于数据质量控制（筛选出 AO/LO 重建失败的分子用于重跑或人工排查）。

---

## 注意事项与已知约束

- 支持的 MO/基组有限（见 `mybasis_verify.json`），对其他基组需补充 `DEFAULT_AUXBASIS` 映射并验证。
- `mkl2fch` 可执行文件为外部依赖（由 MOKIT 提供），必须可执行并在 `PATH` 指定位置。
- LO 重构对数值精度敏感，若多次失败建议放宽 `ORCA_RECON_TOL` 或设为 `SKIP` 并记录问题样本。

---

## 参考文件

- `orca2pyscf.py`, `workflow_orca2pyscf.py`, `validate_ao_lo_consistency.py`
- `ao_lo_error_metrics_notes.tex`（误差定义与解释）
- `mybasis_verify.json`（已检验基组）

## References

- F. Neese, "The ORCA program system", WIREs Computational Molecular Science (2012).
- Q. Sun et al., "PySCF: the Python-based simulations of chemistry framework", WIREs Comput Mol Sci (2018).
- MOKIT utilities used for MKL→FCHK conversion and reading (`mkl2fch`, `fch2py`).


---

*文件生成于项目自动化摘要。若需要包含更多运行示例或将此摘要合并入上游 README，我可以继续更新。*
