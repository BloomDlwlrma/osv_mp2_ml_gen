#!/usr/bin/env python3
# 这段代码的主要作用是为量子化学机器学习训练准备分子对的能量数据，将原始特征文件（ml_features.hdf5）中带方法后缀的数据集（如 pairlist_ccsd、pair_ene_ccsd）提取、重命名，并组装到一个新的 HDF5 输出文件中，同时进行严格的数据校验。
# 具体功能可概括为以下几点：
# 数据抽取与重命名
  # 输入的原始特征文件 raw_fea 里，按分子查找 pairlist_{method} 和 pair_ene_{method} 数据集（例如 pairlist_ccsd、pair_ene_ccsd）。
  # 将它们以去掉后缀的名字（pairlist、pair_ene）存入输出文件 out_pair。
  # 同时会复制相关的元数据，如 nocc、nocc_core、mo_list、pairlist_screened，以及可能存在的 pairlist_offdiag。
# 从已有 pair 文件直接复制
  # 如果提供了 --from-pair 参数，则不再进行“转换”，而是直接从已有的 pair 能量 HDF5 文件中复制指定分子组到输出文件，并对每个组进行同样的校验。这对于合并或筛选分子数据很有用。
# 数据完整性校验
  # _validate_pair_group 函数保证每个分子组中：
  # 必须包含 pairlist 和 pair_ene，且两者第一维长度一致。
  # （默认）必须包含嵌入的元数据 nocc 和 pairlist_screened，且其中的轨道索引在合理范围内（[0, nocc²-1]）。
# 这一校验确保了后续训练使用的 pair 列表和能量是对齐且合法的。
# 命令行接口
  # 支持灵活指定方法（ccsd 或 ccsdt）、分子列表、是否允许缺失元数据、以及是否回退到旧格式（尚未实现）。
  # 典型用法：
  # bash
  # python script.py --raw-fea ml_features.hdf5 --out-pair pair_energy.hdf5 --method ccsd --mols 010295 ...
# 总结： 这是一个数据预处理工具，专门用于从量子化学计算结果中提取和整理“电子对能量”信息，输出为训练神经网络或其他机器学习模型所需的标准 HDF5 格式，同时保证数据的一致性和正确性。
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _molec_name(molid: str) -> str:
    return f"dsgdb9nsd_{molid}"


def _validate_pair_group(grp: h5py.Group, mol: str, allow_missing_meta: bool = False) -> None:
    if "pairlist" not in grp or "pair_ene" not in grp:
        raise RuntimeError(f"{mol}: converted pair group is missing pairlist/pair_ene")

    pairlist = np.asarray(grp["pairlist"][:])
    pair_ene = np.asarray(grp["pair_ene"][:])
    if pairlist.shape[0] != pair_ene.shape[0]:
        raise RuntimeError(
            f"{mol}: pairlist/pair_ene length mismatch ({pairlist.shape[0]} vs {pair_ene.shape[0]})"
        )

    if allow_missing_meta:
        return

    if "nocc" not in grp or "pairlist_screened" not in grp:
        raise RuntimeError(
            f"{mol}: converted pair group must contain embedded nocc and pairlist_screened metadata"
        )

    nocc = int(np.asarray(grp["nocc"][:]).reshape(-1)[0])
    if pairlist.ndim == 1:
        if np.any(pairlist < 0) or np.any(pairlist >= nocc**2):
            raise RuntimeError(f"{mol}: pairlist contains ids outside [0, {nocc**2 - 1}]")
    screened = np.asarray(grp["pairlist_screened"][:]).reshape(-1)
    if np.any(screened < 0) or np.any(screened >= nocc**2):
        raise RuntimeError(f"{mol}: pairlist_screened contains ids outside [0, {nocc**2 - 1}]")


def convert(
    raw_fea: Path,
    out_pair: Path,
    method: str = "ccsd",
    mols: list[str] | None = None,
    fallback: bool = False,
    allow_missing_meta: bool = False,
):
    suffix = f"_{method}"
    with h5py.File(raw_fea, "r") as fin, h5py.File(out_pair, "w") as fout:
        mols_in = list(fin.keys()) if mols is None else [_molec_name(m) for m in mols]
        for mol in mols_in:
            if mol not in fin:
                continue
            grp_in = fin[mol]
            pairlist_name = f"pairlist{suffix}"
            pairene_name = f"pair_ene{suffix}"
            offdiag_name = f"pairlist_offdiag{suffix}"

            if pairlist_name in grp_in and pairene_name in grp_in:
                pl = np.asarray(grp_in[pairlist_name][:])
                pe = np.asarray(grp_in[pairene_name][:])
            else:
                if fallback and "pair_energy" in fin.file:
                    raise RuntimeError("fallback requested but not implemented in converter")
                continue

            g = fout.require_group(mol)
            g.create_dataset("pairlist", data=pl.astype("i8"))
            g.create_dataset("pair_ene", data=pe.astype("f8"))
            if offdiag_name in grp_in:
                g.create_dataset("pairlist_offdiag", data=np.asarray(grp_in[offdiag_name][:]))
            for key in ["nocc", "nocc_core", "mo_list", "pairlist_screened"]:
                if key in grp_in and key not in g:
                    g.create_dataset(key, data=np.asarray(grp_in[key][:]))
            _validate_pair_group(g, mol, allow_missing_meta=allow_missing_meta)


def copy_from_pair(pair_src: Path, out_pair: Path, mols: list[str] | None = None, allow_missing_meta: bool = False):
    with h5py.File(pair_src, "r") as fin, h5py.File(out_pair, "w") as fout:
        for mol in fin.keys() if mols is None else [f"dsgdb9nsd_{m}" for m in mols]:
            if mol not in fin:
                continue
            fin.copy(mol, fout)
            _validate_pair_group(fout[mol], mol, allow_missing_meta=allow_missing_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fea", required=True, help="ml_features.hdf5 with suffixed datasets")
    parser.add_argument("--out-pair", required=True, help="output pair_energy hdf5 for training")
    parser.add_argument("--method", choices=["ccsd", "ccsdt"], default="ccsd")
    parser.add_argument("--mols", nargs="*", help="molecule ids (e.g. 010295)")
    parser.add_argument("--fallback-to-original", action="store_true")
    parser.add_argument("--from-pair", default=None, help="Optional existing pair_energy.hdf5 to copy groups from")
    parser.add_argument("--allow-missing-meta", action="store_true", help="Allow outputs without embedded nocc/pairlist_screened metadata")
    args = parser.parse_args()

    raw = Path(args.raw_fea)
    out = Path(args.out_pair)
    mols = args.mols if args.mols else None
    if args.from_pair:
        copy_from_pair(Path(args.from_pair), out, mols=mols, allow_missing_meta=args.allow_missing_meta)
    else:
        convert(
            raw,
            out,
            method=args.method,
            mols=mols,
            fallback=args.fallback_to_original,
            allow_missing_meta=args.allow_missing_meta,
        )


if __name__ == "__main__":
    main()
