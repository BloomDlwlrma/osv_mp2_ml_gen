'''
解析器: 用正则匹配 ORCA 输出中的 pair correlation 和 triple 行，提供迭代器 iter_pair_corr() 与 iter_triples()，
以及小数据类 PairCorrRecord / TripleRecord 和 molname_from_path()（提取分子 id）。该模块把文本解析成结构化记录，供上层写入 HDF5 使用。
'''
from __future__ import annotations

import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_PAIR_SECTION_TITLE_RE = re.compile(r"^\s*PAIR\s+CORRELATION\s+ENERGIES\s*\(Eh\)\s*$")
_PAIR_INLINE_RE = re.compile(
    r"(?<!\d)(?P<i>\d+)\s+(?P<j>\d+)\s*:\s*"
    r"(?P<ep_guess>[-+0-9.Ee]+)\s+"
    r"(?P<ep_final>[-+0-9.Ee]+)"
)
_TRIPLE_LINE_RE = re.compile(
    r"^\s*(?:(?P<pct>\d+)%\s+done\s+)?"
    r"Triple\s+(?P<i>\d+)\s+(?P<j>\d+)\s+(?P<k>\d+)\s*:"
    r"\s*current eijk=\s*(?P<eijk>[-+0-9.Ee]+)"
    r"\s+ET\(i,j,k\)=\s*(?P<et_ijk>[-+0-9.Ee]+)"
    r"\s+ET=\s*(?P<et_cum>[-+0-9.Ee]+)\s*$"
)


@dataclass(frozen=True)
class PairCorrRecord:
    source_file: str
    i: int
    j: int
    ep_final: float


@dataclass(frozen=True)
class TripleRecord:
    source_file: str
    i: int
    j: int
    k: int
    eijk: float
    et_ijk: float
    et_cum: float


def _expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        if any(ch in raw for ch in "*?["):
            paths.extend(sorted(Path(p) for p in glob.glob(raw, recursive=True)))
        else:
            paths.append(Path(raw))

    seen: set[Path] = set()
    uniq: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    return uniq


def iter_pair_corr(path: Path) -> Iterator[PairCorrRecord]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fin:
            in_section = False
            for line in fin:
                if not in_section:
                    if _PAIR_SECTION_TITLE_RE.match(line):
                        in_section = True
                    continue
                for match in _PAIR_INLINE_RE.finditer(line):
                    i = int(match.group("i"))
                    j = int(match.group("j"))
                    if i > j:
                        i, j = j, i
                    yield PairCorrRecord(
                        source_file=str(path),
                        i=i,
                        j=j,
                        ep_final=float(match.group("ep_final")),
                    )
    except FileNotFoundError:
        return


def iter_triples(path: Path) -> Iterator[TripleRecord]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                match = _TRIPLE_LINE_RE.match(line)
                if not match:
                    continue
                i = int(match.group("i"))
                j = int(match.group("j"))
                if i > j:
                    i, j = j, i
                yield TripleRecord(
                    source_file=str(path),
                    i=i,
                    j=j,
                    k=int(match.group("k")) - 1,
                    eijk=float(match.group("eijk")),
                    et_ijk=float(match.group("et_ijk")),
                    et_cum=float(match.group("et_cum")),
                )
    except FileNotFoundError:
        return


def molname_from_path(path: Path) -> str:
    stem = path.stem
    match = re.match(r'^(dsgdb9nsd_\d{6})(?:_.*)?$', stem)
    if match:
        return match.group(1)
    match = re.match(r'^(.+?_\d{6})(?:_.*)?$', stem)
    if match:
        return match.group(1)
    return stem
