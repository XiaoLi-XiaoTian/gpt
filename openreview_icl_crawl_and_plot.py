#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenReview（ICLR/ICML）ICL 论文统计：title+abstract 检索 + 细分 taxonomy 分类 + 作图

本版修复：
1) 中文显示问题：
   - 更稳健地设置 Matplotlib 字体：font.sans-serif 列表 + font.family='sans-serif'
   - 支持 --font 显式指定字体（mac 常用 "PingFang SC"，Win 常用 "Microsoft YaHei"，Linux 常用 "Noto Sans CJK SC"）
   - 类别标签里带 emoji/图标时，缺少 emoji 字体会出现乱码/方块，因此绘图时默认去掉前缀 emoji。

2) 折线图“只有点没有线”：
   - 常见原因是某些类别只在某一年出现 => 每条线只有一个点，看起来像“没连线”
   - 解决：对 years 做 reindex（缺失年份补 0），并强制 linestyle='-'

3) 仅用已保存数据重画图（不重复抓取）：
   - 使用 --plot_only 直接读取 outdir/icl_papers_filtered.csv 生成图片
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import requests
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

VERSION = "v3.0"


def _display_label(s: str) -> str:
    """Strip leading emoji/symbols so Chinese text renders even if emoji fonts are missing."""
    return re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+\s*", "", str(s)).strip()


# ----------------------------
# 1) ICL 过滤：Title + Abstract
# ----------------------------
ICL_TERMS = [
    r"\bin[- ]context\b",
    r"\bin[- ]context learning\b",
    r"\bICL\b",
    r"\b(in[- ]context) (reason|learn|adapt|generaliz)\w*",
    r"\bmany[- ]shot\b",
    r"\bfew[- ]shot\b",
]
ICL_REGEX = re.compile("|".join(ICL_TERMS), flags=re.IGNORECASE)


# ----------------------------
# 2) Taxonomy
# ----------------------------
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    patterns: Tuple[str, ...]


A_MECH_THEORY = Category(
    "A_mech_theory",
    "🧠 A 机理与理论",
    (
        r"\btheor\w*|\bprovab\w*|\bmechanis\w*|\bdynamics?\b|\binduction head(s)?\b",
        r"\bcircuit(s)?\b|\binterpret\w*|\bassociative memory\b|\bhopfield\b",
        r"\bmeta[- ]learn\w*|\bgradient descent\b|\bimplicit\b",
    ),
)

B_DEMO_SELECT = Category("B_demo_select", "🧩 B1 示例选择/筛选", (
    r"\bexample selection\b|\bdemonstration selection\b|\bexemplar selection\b|\bretrieve demonstrations\b",
    r"\bselect(ing)? (examples|demonstrations|exemplars)\b|\bfilter(ing)? (examples|demonstrations)\b",
))
B_DEMO_ORDER = Category("B_demo_order", "🧩 B2 示例排序/结构化", (
    r"\bprompt ordering\b|\border(ing)? demonstrations\b|\bpermutation\b|\bcurriculum\b",
    r"\bcompose(d)? demonstrations\b|\bstructure(d)? prompt\b",
))
B_COT_REASON = Category("B_cot_reason", "🧩 B3 CoT/多步推理/Many-shot Reasoning", (
    r"\bchain[- ]of[- ]thought\b|\bCoT\b|\bscratchpad\b|\bdeliberat\w*|\bself[- ]consistency\b",
    r"\bmultistep\b|\breason(er|ing)\b|\bmany[- ]shot\b|\bmany[- ]step\b",
))
B_KNN_NONPARAM = Category("B_knn_nonparam", "🧩 B4 近邻/非参数式 ICL 推断", (
    r"\bnearest neighbor\b|\b(k[- ]?nn|kNN)\b|\bnonparametric\b|\bprototype(s)?\b",
    r"\bcalibration[- ]free\b|\bembedding[- ]based inference\b|\bvector database\b",
))
B_MISTAKE_PRINCIPLE = Category("B_mistake_principle", "🧩 B5 从错误学习/原则归纳", (
    r"\bmistake(s)?\b|\berror(s)?\b|\bcounterexample(s)?\b|\bfrom mistakes\b",
    r"\bprinciple learning\b|\brule induction\b|\bself[- ]correction\b",
))
B_CALIB_UQ = Category("B_calib_uq", "🧩 B6 校准/不确定性/拒答", (
    r"\bcalibrat\w*|\buncertaint\w*|\bconfidence\b|\breliabilit\w*\b",
    r"\bselective prediction\b|\babstain\b|\breject option\b",
))

C_CTX_COMPRESS = Category("C_ctx_compress", "📏 C1 上下文压缩/蒸馏/记忆化", (
    r"\bcontext compression\b|\bcompress(ion|ing)?\b|\bdistill(at|ation)\w*\b",
    r"\b(in[- ]context )?autoencoder\b|\bprompt compression\b|\bcontext distillation\b",
))
C_CACHE_EFFIC = Category("C_cache_effic", "📏 C2 推理效率/KV Cache/高效注意力", (
    r"\bkv cache\b|\bkey[- ]value\b|\bcache\b|\bprefill\b|\bthroughput\b|\blatency\b",
    r"\befficient attention\b|\blinear attention\b|\bflash[- ]?attention\b",
))
C_LEN_EXTRAP = Category("C_len_extrap", "📏 C3 长度泛化/长短对齐/长度外推", (
    r"\blength generaliz\w*\b|\blength extrapolat\w*\b|\btrain short\b|\binfer long\b",
    r"\blong[- ]short\b|\bcontext length generaliz\w*\b|\bpositional extrapolat\w*\b",
))

D_TRAIN_ARCH = Category("D_train_arch", "🏗️ D 训练/架构/预训练范式", (
    r"\bpretrain\w*\b|\btraining\b|\barchitecture\b|\bstate space\b|\bxLSTM\b|\bmamba\b",
    r"\bsequence model(ing)?\b|\bmixture of experts\b|\battention variant\b",
))
E_AGENT_PLANNING = Category("E_agent_planning", "🤖 E Agent/规划/工具", (
    r"\bagent(s)?\b|\bplanning\b|\btool use\b|\baction sequence\b|\btrajectory\b|\breasoning and acting\b",
))
F_EVAL_BENCH = Category("F_eval_bench", "📊 F 评测/基准/诊断", (
    r"\bbenchmark\b|\bevaluation\b|\btestbed\b|\bdiagnostic\b|\bprobe\b|\bablation\b|\bmeasure\b",
))
G_SAFETY_PRIVACY = Category("G_safety_privacy", "🛡️ G 安全/隐私/遗忘", (
    r"\bunlearning\b|\bforget(ting)?\b|\bprivacy\b|\bdata leakage\b|\battack\b|\bbackdoor\b",
    r"\bjailbreak\b|\bwatermark\b|\bsafety\b|\brefusal\b",
))

CATEGORY_PRIORITY: List[Category] = [
    G_SAFETY_PRIVACY, F_EVAL_BENCH, E_AGENT_PLANNING,
    C_CTX_COMPRESS, C_CACHE_EFFIC, C_LEN_EXTRAP,
    B_DEMO_SELECT, B_DEMO_ORDER, B_COT_REASON, B_KNN_NONPARAM, B_MISTAKE_PRINCIPLE, B_CALIB_UQ,
    D_TRAIN_ARCH, A_MECH_THEORY
]
DEFAULT_LABEL = "🧺 其他/未归类"


def classify(text: str) -> str:
    for cat in CATEGORY_PRIORITY:
        for p in cat.patterns:
            if re.search(p, text, flags=re.IGNORECASE):
                return cat.label
    return DEFAULT_LABEL


def safe_json(resp: requests.Response) -> Union[dict, list]:
    try:
        return resp.json()
    except Exception:
        snippet = resp.text[:400].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response (status={resp.status_code}). Head: {snippet}")


def extract_notes(payload: Union[dict, list]) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("notes", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


def http_get(baseurl: str, path: str, params: Dict, timeout: int) -> Union[dict, list]:
    url = f"{baseurl.rstrip('/')}{path}"
    headers = {"User-Agent": "ICL-survey-bot/3.0 (requests)", "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return safe_json(r)


def fetch_notes_paginated(baseurl: str, invitation: str, extra_params: Dict, limit: int = 1000, timeout: int = 60, verbose: bool = True) -> List[dict]:
    all_notes: List[dict] = []
    offset = 0
    while True:
        params = {"invitation": invitation, "limit": limit, "offset": offset}
        params.update(extra_params or {})
        payload = http_get(baseurl, "/notes", params=params, timeout=timeout)
        notes = extract_notes(payload)
        if not notes:
            if verbose:
                print(f"    page offset={offset}: 0 notes (stop). total={len(all_notes)}")
            break
        all_notes.extend(notes)
        if verbose:
            print(f"    page offset={offset}: +{len(notes)} notes (total={len(all_notes)})")
        offset += limit
        if offset > 200000:
            break
    return all_notes


def normalize_note(note: dict) -> Tuple[str, str]:
    c = note.get("content", {}) or {}

    # 处理API v2格式（字段可能是dict with 'value'）和API v1格式（直接字符串）
    def extract_value(field):
        if field is None:
            return ""
        if isinstance(field, dict):
            return str(field.get("value", ""))
        return str(field)

    title = extract_value(c.get("title")).strip()
    abstract = extract_value(c.get("abstract")).strip()
    tldr = extract_value(c.get("TL;DR") or c.get("TLDR")).strip()

    if (not abstract) and tldr:
        abstract = tldr
    return title, abstract


def invitation_candidates(conf: str, year: int) -> List[str]:
    venue = f"{conf}.cc/{year}/Conference"
    # API v2 格式（2023+主要使用）
    candidates = [
        f"{venue}/-/Submission",
        f"{venue}/-/Blind_Submission",
        f"{venue}/-/Paper",
    ]
    # API v1/旧格式（作为备选）
    venue_lower = f"{conf}.cc/{year}/conference"
    candidates.extend([
        f"{venue_lower}/-/submission",
        f"{venue_lower}/-/blind_submission",
        f"{venue_lower}/-/Blind_Submission",
    ])
    return candidates


def try_fetch_accepted(conf: str, year: int, verbose: bool, timeout: int) -> Tuple[str, str, int, int, List[dict]]:
    venueid = f"{conf}.cc/{year}/Conference"
    # API v2 endpoint优先（2023+主要使用）
    baseurls = ["https://api2.openreview.net", "https://api.openreview.net"]
    invs = invitation_candidates(conf, year)
    select = "id,number,content.title,content.abstract,content.TL;DR,content.TLDR,content.venueid,content.venue"

    last_errs = []
    for base in baseurls:
        for inv in invs:
            if verbose:
                print(f"[{conf} {year}] probing base={base} invitation={inv}", flush=True)

            # 尝试1: 通过content.venueid过滤accepted papers（API v2推荐方式）
            extra = {"select": select, "content.venueid": venueid}
            try:
                acc_notes = fetch_notes_paginated(base, inv, extra_params=extra, limit=1000, timeout=timeout, verbose=verbose)
                if acc_notes:
                    if verbose:
                        print(f"  ✓ found {len(acc_notes)} accepted papers via content.venueid", flush=True)
                    return base, inv, -1, len(acc_notes), acc_notes
            except Exception as e:
                last_errs.append(f"{base} {inv} (venueid filter): {str(e)[:200]}")
                if verbose:
                    print(f"  !! venueid filter failed: {e}", flush=True)

            # 尝试2: 获取所有submissions，手动过滤accepted（备选方案）
            extra2 = {"select": select}
            try:
                subs = fetch_notes_paginated(base, inv, extra_params=extra2, limit=1000, timeout=timeout, verbose=False)
                if subs:
                    # 手动过滤accepted papers
                    accepted = []
                    for note in subs:
                        content = note.get("content", {}) or {}
                        note_venueid = content.get("venueid", "")
                        venue = content.get("venue", "")
                        # 检查是否为accepted paper
                        if (note_venueid == venueid or
                            (venue and ("accept" in venue.lower() or f"{conf} {year}" in venue))):
                            accepted.append(note)

                    if accepted:
                        if verbose:
                            print(f"  ✓ found {len(accepted)} accepted papers from {len(subs)} submissions (manual filter)", flush=True)
                        return base, inv, len(subs), len(accepted), accepted
                    else:
                        last_errs.append(f"{base} {inv}: has submissions={len(subs)} but no accepted papers found")
                        if verbose:
                            print(f"  !! {len(subs)} submissions but no accepted papers", flush=True)
            except Exception as e:
                last_errs.append(f"{base} {inv} (all submissions): {str(e)[:200]}")
                if verbose:
                    print(f"  !! fetch all submissions failed: {e}", flush=True)

    raise RuntimeError("Could not fetch any submissions/accepted. Errors (last 12):\n" + "\n".join(last_errs[-12:]))


def set_cjk_font(preferred_font: str = None):
    candidates = [
        "PingFang SC", "Heiti SC", "Songti SC",
        "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "WenQuanYi Zen Hei",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}

    chosen = preferred_font if preferred_font else None
    if not chosen:
        for name in candidates:
            if name in available:
                chosen = name
                break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        print(
            "[WARN] No CJK font found by name. Chinese may not render. "
            "Install a CJK font (e.g., PingFang SC / Microsoft YaHei / Noto Sans CJK SC) "
            "or run with --font '<Font Name>'.",
            file=sys.stderr,
            flush=True,
        )

    plt.rcParams["axes.unicode_minus"] = False


def _wrap(s: str, width: int = 18) -> str:
    out, cur = [], ""
    for ch in s:
        cur += ch
        if len(cur) >= width and ch not in (" ", "：", ":", "/", "-"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return "\n".join(out)


def plot_donut_pie(df: pd.DataFrame, outpath: str, title: str, subtitle: str):
    counts = df["category"].value_counts()
    labels = counts.index.tolist()
    disp_labels = [_display_label(x) for x in labels]
    values = counts.values.tolist()

    fig = plt.figure(figsize=(13.5, 9), dpi=220)
    ax = fig.add_subplot(111)

    wedges, _, autotexts = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
    )
    for t in autotexts:
        t.set_fontsize(10)

    ax.set_title(title, fontsize=18, pad=16)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=11)

    total = int(counts.sum())
    ax.text(0, 0, f"ICL相关\n{total} 篇", ha="center", va="center", fontsize=16, fontweight="bold")

    legend_labels = [f"{_wrap(lab)}\n{val}篇" for lab, val in zip(disp_labels, values)]
    ax.legend(
        wedges, legend_labels,
        title="研究方向（规则分类）",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=10.5, title_fontsize=12,
        labelspacing=0.8, handlelength=1.2,
    )

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_trend(df: pd.DataFrame, outpath: str, title: str, topk: int, years: List[int]):
    pivot = (
        df.groupby(["year", "category"]).size().reset_index(name="count")
          .pivot(index="year", columns="category", values="count")
          .fillna(0).astype(int).sort_index()
    )

    years = [int(y) for y in years]
    pivot = pivot.reindex(years, fill_value=0)

    totals = pivot.sum(axis=0).sort_values(ascending=False)
    keep = totals.head(topk).index.tolist()
    pivot_small = pivot[keep].copy() if keep else pivot.copy()
    if len(totals) > topk:
        pivot_small["🧺 其他（长尾）"] = pivot.drop(columns=keep).sum(axis=1)

    fig = plt.figure(figsize=(15.5, 7.8), dpi=220)
    ax = fig.add_subplot(111)

    for col in pivot_small.columns:
        ax.plot(
            pivot_small.index,
            pivot_small[col],
            marker="o",
            linestyle="-",
            linewidth=2,
            label=_display_label(col),
        )

    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("论文数量（篇）", fontsize=12)
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", alpha=0.35)

    yearly_total = pivot.sum(axis=1)
    ymax = max(1, pivot_small.max(axis=1).max())
    for x, y in yearly_total.items():
        ax.annotate(f"总计 {int(y)}", (x, ymax), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)

    ax.set_ylim(0, max(2, ymax + 3))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--confs", nargs="+", default=["ICLR", "ICML"])
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--quiet", action="store_true", help="减少日志输出")
    ap.add_argument("--plot_only", action="store_true", help="仅从已保存的 CSV 生成图像（不重新抓取）")
    ap.add_argument("--data_csv", default=None, help="plot_only 模式下使用的 CSV 路径（默认 outdir/icl_papers_filtered.csv）")
    ap.add_argument("--font", default=None, help="指定 Matplotlib 字体名称（解决中文显示问题，例如 'PingFang SC'）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_cjk_font(args.font)

    verbose = not args.quiet
    print(f"Running {os.path.basename(__file__)} {VERSION}", flush=True)

    if args.plot_only:
        csv_path = args.data_csv or os.path.join(args.outdir, "icl_papers_filtered.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] plot_only=True but CSV not found: {csv_path}", file=sys.stderr)
            return
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in ("year", "category"):
            if col not in df.columns:
                print(f"[ERROR] CSV missing required column: {col}", file=sys.stderr)
                return

        pie_out = os.path.join(args.outdir, "icl_pie_donut_refined.png")
        trend_out = os.path.join(args.outdir, "icl_trend_lines_refined.png")

        confs_str = " & ".join(args.confs)
        year_min, year_max = min(args.years), max(args.years)
        subtitle = "口径：OpenReview title+abstract（已抓取并保存）；分类：规则匹配（可复现）"

        plot_donut_pie(
            df, pie_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向占比（细分 taxonomy，title+abstract）",
            subtitle=subtitle
        )
        plot_trend(
            df, trend_out,
            title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向发文趋势（细分 taxonomy，title+abstract）",
            topk=args.topk,
            years=args.years,
        )

        print("\n[OK] Plot regenerated from saved CSV:")
        print(" -", pie_out)
        print(" -", trend_out)
        return

    rows: List[Dict] = []
    meta_rows: List[Dict] = []

    for conf in args.confs:
        for year in args.years:
            try:
                base, inv, sub_n, acc_n, acc_notes = try_fetch_accepted(conf, year, verbose=verbose, timeout=args.timeout)
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": base, "invitation": inv,
                    "submissions": sub_n, "accepted": acc_n
                })
                if verbose:
                    print(f"[{conf} {year}] ✅ accepted fetched: {acc_n} (base={base}, inv={inv})", flush=True)

                for n in tqdm(acc_notes, disable=not verbose, desc=f"{conf}-{year} filter"):
                    title, abstract = normalize_note(n)
                    text = f"{title}\n{abstract}"
                    if not ICL_REGEX.search(text):
                        continue
                    cat = classify(text)
                    rows.append({
                        "conf": conf, "year": year,
                        "title": title, "abstract": abstract,
                        "category": cat
                    })

                if verbose:
                    print(f"[{conf} {year}] ICL matched: {sum(1 for r in rows if r['conf']==conf and r['year']==year)}", flush=True)

            except Exception as e:
                meta_rows.append({
                    "conf": conf, "year": year,
                    "baseurl": "", "invitation": "",
                    "submissions": 0, "accepted": 0,
                    "error": str(e)[:400]
                })
                print(f"[{conf} {year}] ❌ FAILED: {e}", file=sys.stderr, flush=True)

    meta = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.outdir, "fetch_meta.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df_path = os.path.join(args.outdir, "icl_papers_filtered.csv")
    df.to_csv(df_path, index=False, encoding="utf-8-sig")

    if df.empty:
        print("\nNo papers matched ICL_REGEX under the accepted set.", flush=True)
        print("Try relaxing ICL_TERMS OR verify that accepted set is fetched correctly.", flush=True)
        print(f"See: {meta_path}", flush=True)
        return

    pie_out = os.path.join(args.outdir, "icl_pie_donut_refined.png")
    trend_out = os.path.join(args.outdir, "icl_trend_lines_refined.png")

    confs_str = " & ".join(args.confs)
    year_min, year_max = min(args.years), max(args.years)
    subtitle = "口径：OpenReview title+abstract；accepted 过滤优先用 content.venueid；分类：规则匹配（可复现）"

    plot_donut_pie(
        df, pie_out,
        title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向占比（细分 taxonomy，title+abstract）",
        subtitle=subtitle
    )
    plot_trend(
        df, trend_out,
        title=f"{confs_str}（{year_min}–{year_max}）ICL 相关论文：研究方向发文趋势（细分 taxonomy，title+abstract）",
        topk=args.topk,
        years=args.years,
    )

    print("\nSaved:", flush=True)
    print(" -", df_path)
    print(" -", meta_path)
    print(" -", pie_out)
    print(" -", trend_out)


if __name__ == "__main__":
    main()
