# -*- coding: utf-8 -*-
"""
step6_calc_metrics.py
=====================
基于 SR 显著性图与任务区域标注，计算三种权重条件（正确 / 均匀 / 反向）下的
区域级指标：AUC、NSS、CC、命中（NSS >= 1.0），以及相对增强比 RER。

输入（相对 repo 根目录）：
  data/scenarios.csv          -- 30 个场景定义（UTF-8-sig）
  data/annotations.csv        -- 全部功能区域坐标 + 权重（GBK）
  output/<base>_saliency.jpg  -- step4 产出的 SR 显著性图（灰度）

输出：
  results/step6_metrics_by_scenario.csv   -- 逐场景 × 三条件明细
  results/step6_summary.csv               -- 三条件命中率 / RER 均值±SD 汇总

权重条件：
  Correct  -- annotations.csv 中原始权重
  Uniform  -- 所有区域权重 = 1.0（等于不做加权）
  Reversed -- 每个区域权重取倒数 1/w（保持相对间距、反转优先级）

注意：RER 分母固定为原始 SR 全局均值（论文 Eq.14），不随权重变化。
"""

import os
import csv
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# ========== 路径（相对 repo 根目录） ==========
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SALIENCY_DIR = REPO_ROOT / "output"          # step4 产物
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS_CSV = DATA_DIR / "scenarios.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
DETAIL_CSV = OUT_DIR / "step6_metrics_by_scenario.csv"
SUMMARY_CSV = OUT_DIR / "step6_summary.csv"
# ============================================

HIT_THRESHOLD = 1.0  # NSS >= 1.0 记为命中


def read_annotations(path, encoding):
    """返回 coord[(fn,region)] = (x1,y1,x2,y2) 和 weights[(fn,region)] = float。"""
    coord, weights = {}, {}
    with open(path, "r", encoding=encoding) as f:
        for row in csv.DictReader(f):
            key = (row["文件名"], row["区域名称"])
            coord[key] = (
                int(float(row["x1"])), int(float(row["y1"])),
                int(float(row["x2"])), int(float(row["y2"])),
            )
            weights[key] = float(row["权重"])
    return coord, weights


def build_regions_by_file(coord, weights):
    """按文件名分组：{fn: [(x1,y1,x2,y2,w), ...]}。"""
    out = {}
    for key, (x1, y1, x2, y2) in coord.items():
        fn = key[0]
        out.setdefault(fn, []).append((x1, y1, x2, y2, weights[key]))
    return out


def weight_matrix(shape, regions, weight_overrides=None):
    """根据区域列表构造权重矩阵。weight_overrides: {(fn,region): w} 用于均匀/反向。"""
    h, w = shape
    wm = np.ones((h, w), dtype=np.float32)
    for x1, y1, x2, y2, w_orig in regions:
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        wm[y1:y2+1, x1:x2+1] = w_orig
    return wm


def crop_box(sal, box):
    h, w = sal.shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    return sal[y1:y2+1, x1:x2+1]


def nss(sal, box):
    """目标区域均值相对全局标准化后的 z-score。"""
    gm, gs = sal.mean(), sal.std()
    if gs == 0:
        return 0.0
    return float((crop_box(sal, box).mean() - gm) / gs)


def auc(sal, mask):
    try:
        return float(roc_auc_score(mask.flatten(), sal.flatten()))
    except ValueError:
        return 0.5


def cc(sal, mask):
    if sal.std() == 0 or mask.std() == 0:
        return 0.0
    return float(pearsonr(sal.flatten(), mask.flatten())[0])


def main():
    # 读 scenarios
    with open(SCENARIOS_CSV, "r", encoding="utf-8-sig") as f:
        scenarios = list(csv.DictReader(f))

    # 读 annotations（GBK）
    coord_corr, w_corr = read_annotations(ANNOTATIONS_CSV, "gbk")
    regions_corr = build_regions_by_file(coord_corr, w_corr)

    # 派生均匀 / 反向权重
    w_uniform = {k: 1.0 for k in w_corr}
    w_reversed = {k: (1.0 / v if v > 0 else 0.0) for k, v in w_corr.items()}
    regions_uniform = build_regions_by_file(coord_corr, w_uniform)
    regions_reversed = build_regions_by_file(coord_corr, w_reversed)

    rows = []
    for sc in scenarios:
        sid = sc["场景编号"]; fn = sc["图片文件名"]; target = sc["目标区域名称"]
        key = (fn, target)
        if key not in coord_corr:
            print(f"[warn] 缺标注 {key}，跳过")
            continue
        box = coord_corr[key]
        base = os.path.splitext(fn)[0]
        sal_path = SALIENCY_DIR / f"{base}_saliency.jpg"
        if not sal_path.exists():
            print(f"[warn] 缺显著性图 {sal_path}，跳过")
            continue
        sal = cv2.imread(str(sal_path), cv2.IMREAD_GRAYSCALE)
        if sal is None:
            continue
        h, w = sal.shape
        # 任务掩膜
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        mask[y1:y2+1, x1:x2+1] = 1

        sal_f = sal.astype(np.float32)
        # 三种加权图
        wm_corr = weight_matrix((h, w), regions_corr[fn])
        wm_uni = weight_matrix((h, w), regions_uniform[fn])
        wm_rev = weight_matrix((h, w), regions_reversed[fn])
        sal_corr = np.clip(sal_f * wm_corr, 0, 255).astype(np.float32)
        sal_uni = np.clip(sal_f * wm_uni, 0, 255).astype(np.float32)
        sal_rev = np.clip(sal_f * wm_rev, 0, 255).astype(np.float32)

        # RER：分母固定为原始 SR 全局均值；分子权重 * 原始目标区域均值
        gm = sal_f.mean()
        tm = sal_f[mask == 1].mean()
        rer_corr = w_corr[key] * tm / gm if gm > 0 else np.nan
        rer_uni = 1.0 * tm / gm if gm > 0 else np.nan
        rer_rev = (w_reversed[key]) * tm / gm if gm > 0 else np.nan

        def block(sal_img):
            return {
                "AUC": round(auc(sal_img, mask), 4),
                "NSS": round(nss(sal_img, box), 4),
                "CC": round(cc(sal_img, mask), 4),
                "Hit": int(nss(sal_img, box) >= HIT_THRESHOLD),
            }

        b_uw = block(sal_f)
        b_c = block(sal_corr); b_c["RER"] = round(rer_corr, 4)
        b_u = block(sal_uni); b_u["RER"] = round(rer_uni, 4)
        b_r = block(sal_rev); b_r["RER"] = round(rer_rev, 4)

        rows.append({
            "scenario": sid, "file": fn, "target": target,
            "weight": w_corr[key],
            "unweighted_AUC": b_uw["AUC"], "unweighted_NSS": b_uw["NSS"],
            "unweighted_CC": b_uw["CC"], "unweighted_hit": b_uw["Hit"],
            "correct_AUC": b_c["AUC"], "correct_NSS": b_c["NSS"],
            "correct_CC": b_c["CC"], "correct_hit": b_c["Hit"], "correct_RER": b_c["RER"],
            "uniform_AUC": b_u["AUC"], "uniform_NSS": b_u["NSS"],
            "uniform_CC": b_u["CC"], "uniform_hit": b_u["Hit"], "uniform_RER": b_u["RER"],
            "reversed_AUC": b_r["AUC"], "reversed_NSS": b_r["NSS"],
            "reversed_CC": b_r["CC"], "reversed_hit": b_r["Hit"], "reversed_RER": b_r["RER"],
        })

    # 写明细
    fields = list(rows[0].keys())
    with open(DETAIL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    # 汇总
    n = len(rows)
    def rate(prefix, key):
        c = sum(1 for r in rows if r[prefix + "_" + key] == 1)
        return c, n, round(100.0 * c / n, 1)
    def mean_sd(prefix, key):
        vals = [r[prefix + "_" + key] for r in rows]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)

    summary = []
    for prefix, label in [("unweighted", "Unweighted SR"),
                          ("correct", "Correct weights"),
                          ("uniform", "Uniform weights"),
                          ("reversed", "Reversed weights")]:
        c, ntot, pct = rate(prefix, "hit")
        auc_m, auc_sd = mean_sd(prefix, "AUC")
        nss_m, nss_sd = mean_sd(prefix, "NSS")
        cc_m, cc_sd = mean_sd(prefix, "CC")
        row = {"condition": label, "n": ntot,
               "hits": c, "hit_rate_pct": pct,
               "AUC_mean": auc_m, "AUC_sd": auc_sd,
               "NSS_mean": nss_m, "NSS_sd": nss_sd,
               "CC_mean": cc_m, "CC_sd": cc_sd}
        if prefix in ("correct", "uniform", "reversed"):
            rer_m, rer_sd = mean_sd(prefix, "RER")
            row["RER_mean"] = rer_m; row["RER_sd"] = rer_sd
        else:
            row["RER_mean"] = ""; row["RER_sd"] = ""
        summary.append(row)

    sfields = list(summary[0].keys())
    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sfields)
        wr.writeheader()
        wr.writerows(summary)

    print(f"[done] 场景数 = {n}")
    for s in summary:
        print(f"  {s['condition']:18s} hit={s['hits']}/{s['n']} ({s['hit_rate_pct']}%)  "
              f"AUC={s['AUC_mean']}±{s['AUC_sd']}  NSS={s['NSS_mean']}±{s['NSS_sd']}  "
              f"RER={s['RER_mean']}±{s['RER_sd']}")
    print(f"明细 -> {DETAIL_CSV}")
    print(f"汇总 -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
