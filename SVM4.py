#KTH SF2935
#Name: Cecil Knudsen
#SVM for Vehicle re-id 
import json, time, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# CONFIG
CONFIG = {
    "DATA_ROOT": r"C:\Users\cecil\Documents\dataset\archive\VeRi",
    "IMG_SIZE":  [128, 128],   # (H, W) 
    "SEED": 42,
    "OUT_DIR": "outputs",
    "USE_CAMERA_HOLDOUT": False,   
    "HOLDOUT_CAM": "c002",

    # Bootstrap
    "BOOT_N": 1000,
    "CI": 95,

    # SVM/PCA  settings
    "SVM_USE_RGB": True,        # True = RGB pixels (flattened 3*H*W), False = grayscale
    "PCA_DIM": 96,
    "MAX_PER_ID_TRAIN": None,   # None = ALL training images

    # Baseline toggles
    "RUN_PCA_ONLY": True,
    "RUN_LINEAR_SGD": True,
    "RUN_RBF_APPROX": True,
    "RFF_COMPONENTS": 512,
    "RFF_GAMMA": None           # if None -> 1.0 / PCA_DIM
}

OUT_FILE = "veri_svm_results.json"
BATCH_DECISION = 4096


# Helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_outdir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def save_json(path: str | Path, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

def parse_vid_from_filename(path: str) -> str:
    return Path(path).stem.split("_")[0]

def parse_cam_from_filename(path: str) -> str:
    toks = Path(path).stem.split("_")
    return toks[1] if len(toks) > 1 else "c000"

def camera_split_train(paths: List[Path], holdout_cam: str) -> Tuple[List[Path], List[Path]]:
    tr, va = [], []
    for p in paths:
        cam = parse_cam_from_filename(str(p))
        (va if cam == holdout_cam else tr).append(p)
    return tr, va

def read_list_txt(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_veri_eval_lists(root: Path):
    q_names = read_list_txt(root / "name_query.txt")
    g_names = read_list_txt(root / "name_test.txt")
    gt_map, jk_map = {}, {}
    for i, line in enumerate(read_list_txt(root / "gt_index.txt")):
        if not line: gt_map[i] = set(); continue
        idxs = [int(tok) - 1 for tok in line.split()]
        gt_map[i] = {g_names[j] for j in idxs if 0 <= j < len(g_names)}
    for i, line in enumerate(read_list_txt(root / "jk_index.txt")):
        if not line: jk_map[i] = set(); continue
        idxs = [int(tok) - 1 for tok in line.split()]
        jk_map[i] = {g_names[j] for j in idxs if 0 <= j < len(g_names)}
    return q_names, g_names, gt_map, jk_map

# Evaluation
def cmc_map(q_emb, q_names, g_emb, g_names, gt_map, jk_map, topk=(1,5,10)):
    """
    VeRi protocol evaluator:
      - removes junk (same-camera etc.)
      - returns:
          metrics (means),
          ap_per_q (length Q),
          cmc_ind (Q x K binary),
          prec_at_k (Q x K in [0,1])
    """
    Q = normalize(q_emb, axis=1)
    G = normalize(g_emb, axis=1)
    dists = 1.0 - (Q @ G.T)
    order = np.argsort(dists, axis=1)

    K = len(topk)
    ap_per_q   = np.zeros(Q.shape[0], dtype=np.float64)
    cmc_ind    = np.zeros((Q.shape[0], K), dtype=np.float64)  # 1 if first correct within k
    prec_at_k  = np.zeros((Q.shape[0], K), dtype=np.float64)  # precision among first k

    for qi in tqdm(range(Q.shape[0]), desc="eval: VeRi (AP/CMC/Precision)", unit="q"):
        ranked = order[qi]
        gt_set   = gt_map.get(qi, set())
        junk_set = jk_map.get(qi, set())

        good = np.array([g_names[j] in gt_set   for j in ranked], dtype=bool)
        junk = np.array([g_names[j] in junk_set for j in ranked], dtype=bool)

        keep = ~junk
        good_kept = good[keep]  # boolean vector aligned to kept ranks

        # CMC indicators
        if good_kept.any():
            first_pos = int(np.argmax(good_kept))
            for kidx, k in enumerate(topk):
                cmc_ind[qi, kidx] = 1.0 if first_pos < k else 0.0

        if good_kept.sum() == 0:
            ap_per_q[qi] = 0.0
        else:
            hits = 0
            precisions = []
            for r, is_good in enumerate(good_kept, start=1):
                if is_good:
                    hits += 1
                    precisions.append(hits / r)
            ap_per_q[qi] = float(np.mean(precisions))


        for kidx, k in enumerate(topk):
            topk_slice = good_kept[:k]
            if topk_slice.size == 0:
                prec_at_k[qi, kidx] = 0.0
            else:
                prec_at_k[qi, kidx] = float(topk_slice.mean())  # fraction of true among first k

    # Means
    metrics = {
        "mAP": float(ap_per_q.mean()),
        **{f"CMC@{k}": float(cmc_ind[:, idx].mean()) for idx, k in enumerate(topk)},
        **{f"Precision@{k}": float(prec_at_k[:, idx].mean()) for idx, k in enumerate(topk)},
        **{f"FP@{k}": float(1.0 - prec_at_k[:, idx].mean()) for idx, k in enumerate(topk)},
        "Correct@1": float(cmc_ind[:, 0].mean()),
        "FP@1_overall": float(1.0 - cmc_ind[:, 0].mean()),
    }
    return metrics, ap_per_q, cmc_ind, prec_at_k

def bootstrap(ap_per_q: np.ndarray,
                       cmc_ind: np.ndarray,
                       prec_at_k: np.ndarray,
                       n_boot: int,
                       alpha: float,
                       seed: int):
    """
    Query-level bootstrap CIs for:
      - mAP
      - CMC@k
      - Precision@k and FP@k = 1 - Precision@k
    Returns dict of CI tuples.
    """
    rng = np.random.default_rng(seed)
    Q = ap_per_q.shape[0]
    K = cmc_ind.shape[1]
    # stats matrix per bootstrap: [mAP, C1..CK, P1..PK]
    stats = np.empty((n_boot, 1 + K + K), dtype=np.float64)
    for b in tqdm(range(n_boot), desc="Bootstrapping CIs", unit="boot", leave=False):
        idx = rng.integers(0, Q, size=Q)   # resample queries
        stats[b, 0]        = ap_per_q[idx].mean()
        stats[b, 1:1+K]    = cmc_ind[idx].mean(axis=0)
        stats[b, 1+K:1+K+K]= prec_at_k[idx].mean(axis=0)

    low  = np.percentile(stats, 100*alpha/2, axis=0)
    high = np.percentile(stats, 100*(1 - alpha/2), axis=0)

    out = {"mAP_CI": (float(low[0]), float(high[0]))}
    for i in range(K):
        out[f"CMC@{[1,5,10,15,20][:K][i]}_CI"] = (float(low[1+i]), float(high[1+i]))
    for i in range(K):
        out[f"Precision@{[1,5,10,15,20][:K][i]}_CI"] = (float(low[1+K+i]), float(high[1+K+i]))
        
        # FP@k CI is 1 - Precision@k → invert the interval endpoints
        lo_p, hi_p = low[1+K+i], high[1+K+i]
        out[f"FP@{[1,5,10,15,20][:K][i]}_CI"] = (float(1.0 - hi_p), float(1.0 - lo_p))
    return out


# Pipeline (data + features)
def load_split(data_root: Path):
    tr = data_root / "image_train"
    qu = data_root / "image_query"
    ga = data_root / "image_test"
    assert tr.exists() and qu.exists() and ga.exists(), f"Missing VeRi dirs in {data_root}"
    return list_images(tr), list_images(qu), list_images(ga)

def load_and_resize(path: str, size, use_rgb: bool) -> np.ndarray:
    img = Image.open(path).convert("RGB" if use_rgb else "L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if use_rgb:
        return arr.reshape(-1)         
    else:
        return arr.flatten()             

def build_matrix(paths: List[Path], size=(112,112), max_per_id=None, stage="", use_rgb=False):
    by_id: Dict[str, List[Path]] = {}
    for p in paths:
        vid = parse_vid_from_filename(str(p))
        by_id.setdefault(vid, []).append(p)
    selected: List[Tuple[str,str]] = []
    if max_per_id is None:
        for vid, lst in by_id.items():
            selected.extend((str(p), vid) for p in lst)
    else:
        for vid, lst in by_id.items():
            selected.extend((str(p), vid) for p in lst[:max_per_id])
    pix_per_img = size[0]*size[1]*(3 if use_rgb else 1)
    X = np.zeros((len(selected), pix_per_img), dtype=np.float32)
    y, names = [], []
    for i, (path, vid) in enumerate(tqdm(selected, desc=f"{stage}: load images", unit="img")):
        X[i] = load_and_resize(path, size, use_rgb)
        y.append(vid); names.append(Path(path).name)
    return X, np.array(y), names

def decision_function_batched(clf, X, desc="decision", batch=BATCH_DECISION):
    n = X.shape[0]; outs = []
    for i in tqdm(range(0, n, batch), desc=desc, unit="batch"):
        part = clf.decision_function(X[i:i+batch])
        if np.ndim(part) == 1: part = part[:, None]
        outs.append(part)
    return np.vstack(outs)

def sgd_linear_embeddings(Xtr, ytr, Xq, Xg):
    clf = SGDClassifier(loss="hinge", alpha=1e-4,
                        max_iter=30, early_stopping=True,
                        random_state=CONFIG["SEED"])
    tqdm.write("fit SGD (linear hinge) ...")
    t0 = time.time(); clf.fit(Xtr, ytr)
    tqdm.write(f"SGD fit {time.time()-t0:.1f}s")
    q_emb = decision_function_batched(clf, Xq, desc="linear: query")
    g_emb = decision_function_batched(clf, Xg, desc="linear: gallery")
    return q_emb, g_emb

def rbf_approx_embeddings(Xtr, ytr, Xq, Xg, n_comp, gamma):
    rff = RBFSampler(gamma=gamma, n_components=n_comp, random_state=CONFIG["SEED"])
    tqdm.write("RFF fit/transform ...")
    t0 = time.time(); Ztr = rff.fit_transform(Xtr); Zq = rff.transform(Xq); Zg = rff.transform(Xg)
    tqdm.write(f"RFF done {time.time()-t0:.1f}s (Ztr={Ztr.shape})")
    return sgd_linear_embeddings(Ztr, ytr, Zq, Zg)


# Calibration plot 
def plot_reliability(confs, correct, out_path: Path, title="Calibration"):
    prob_true, prob_pred = calibration_curve(correct, confs, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot([0,1],[0,1],"--",linewidth=1)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


def main():
    set_seed(CONFIG["SEED"])
    ensure_outdir(CONFIG["OUT_DIR"])

    root = Path(CONFIG["DATA_ROOT"])
    # Require protocol files
    proto = ["gt_index.txt","jk_index.txt","name_query.txt","name_test.txt"]
    if not all((root/fn).exists() for fn in proto):
        raise RuntimeError("VeRi protocol files not found in DATA_ROOT.")

    tr_all, q_paths, g_paths = load_split(root)
    if CONFIG["USE_CAMERA_HOLDOUT"]:
        tr_paths, _ = camera_split_train(tr_all, CONFIG["HOLDOUT_CAM"])
    else:
        tr_paths = tr_all

    use_rgb = bool(CONFIG.get("SVM_USE_RGB", False))

    # Data matrices
    Xtr_px, ytr, _   = build_matrix(tr_paths, size=tuple(CONFIG["IMG_SIZE"]),
                                    max_per_id=CONFIG["MAX_PER_ID_TRAIN"], stage="train",
                                    use_rgb=use_rgb)
    Xq_px,  yq, qn   = build_matrix(q_paths, size=tuple(CONFIG["IMG_SIZE"]), stage="query",
                                    use_rgb=use_rgb)
    Xg_px,  yg, gn   = build_matrix(g_paths, size=tuple(CONFIG["IMG_SIZE"]), stage="gallery",
                                    use_rgb=use_rgb)

    # Scale + PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr_px)
    Xq_s  = scaler.transform(Xq_px)
    Xg_s  = scaler.transform(Xg_px)

    n_comp = min(CONFIG["PCA_DIM"], Xtr_s.shape[0], Xtr_s.shape[1])
    if n_comp < CONFIG["PCA_DIM"]:
        tqdm.write(f"PCA_DIM clipped to {n_comp}.")
    pca = PCA(n_components=n_comp, svd_solver="randomized", whiten=True, random_state=CONFIG["SEED"])
    Xtr = pca.fit_transform(Xtr_s); Xq = pca.transform(Xq_s); Xg = pca.transform(Xg_s)

    #Calibration
    try:
        rng = np.random.RandomState(CONFIG["SEED"])
        by_id = {}
        for i, vid in enumerate(ytr):
            by_id.setdefault(vid, []).append(i)
        tr_idx, va_idx = [], []
        for lst in by_id.values():
            lst = np.array(lst); rng.shuffle(lst)
            n_val = max(1, int(0.1 * len(lst)))
            va_idx.extend(lst[:n_val]); tr_idx.extend(lst[n_val:])

        clf_log = SGDClassifier(
            loss="log_loss", alpha=1e-4, max_iter=2000,
            early_stopping=True, random_state=CONFIG["SEED"]
        )
        clf_log.fit(Xtr[tr_idx], ytr[tr_idx])

        prob = clf_log.predict_proba(Xtr[va_idx])       
        conf = prob.max(axis=1)                      
        pred_idx = prob.argmax(axis=1)                    
        pred_labels = clf_log.classes_[pred_idx]           
        corr = (pred_labels == ytr[va_idx]).astype(np.int32)

        plot_reliability(conf, corr,
                         Path(CONFIG["OUT_DIR"]) / "svm_calibration.png",
                         "SVM Calibration (closed-set)")
    except Exception as e:
        tqdm.write(f"Calibration skipped: {e}")

    
    q_ref, g_ref, gt_map, jk_map = load_veri_eval_lists(root)
    pos_q = {n:i for i,n in enumerate(qn)}
    pos_g = {n:i for i,n in enumerate(gn)}
    q_idx = np.array([pos_q[n] for n in q_ref], dtype=np.int64)
    g_idx = np.array([pos_g[n] for n in g_ref], dtype=np.int64)

    results = {}
    alpha = 1.0 - CONFIG["CI"] / 100.0

    # PCA-only
    if CONFIG["RUN_PCA_ONLY"]:
        res_pca, ap_pca, cmc_pca, prec_pca = cmc_map(
            Xq[q_idx], q_ref, Xg[g_idx], g_ref, gt_map, jk_map
        )
        ci_pca = bootstrap(np.array(ap_pca), cmc_pca, prec_pca,
                                    n_boot=CONFIG["BOOT_N"], alpha=alpha, seed=CONFIG["SEED"])
        res_pca["mAP_CI95"]   = list(ci_pca["mAP_CI"])
        res_pca["CMC@1_CI95"] = list(ci_pca["CMC@1_CI"])
        res_pca["CMC@5_CI95"] = list(ci_pca["CMC@5_CI"])
        res_pca["CMC@10_CI95"]= list(ci_pca["CMC@10_CI"])
        res_pca["Precision@1_CI95"] = list(ci_pca["Precision@1_CI"])
        res_pca["Precision@5_CI95"] = list(ci_pca["Precision@5_CI"])
        res_pca["Precision@10_CI95"]= list(ci_pca["Precision@10_CI"])
        res_pca["FP@1_CI95"] = list(ci_pca["FP@1_CI"])
        res_pca["FP@5_CI95"] = list(ci_pca["FP@5_CI"])
        res_pca["FP@10_CI95"]= list(ci_pca["FP@10_CI"])
        results["pca_only"] = res_pca
        tqdm.write("PCA-only:\n" + json.dumps(res_pca, indent=2))

    # Linear SVM
    if CONFIG["RUN_LINEAR_SGD"]:
        q_lin, g_lin = sgd_linear_embeddings(Xtr, ytr, Xq, Xg)
        res_lin, ap_lin, cmc_lin, prec_lin = cmc_map(
            q_lin[q_idx], q_ref, g_lin[g_idx], g_ref, gt_map, jk_map
        )
        ci_lin = bootstrap(np.array(ap_lin), cmc_lin, prec_lin,
                                    n_boot=CONFIG["BOOT_N"], alpha=alpha, seed=CONFIG["SEED"])
        res_lin["mAP_CI95"]   = list(ci_lin["mAP_CI"])
        res_lin["CMC@1_CI95"] = list(ci_lin["CMC@1_CI"])
        res_lin["CMC@5_CI95"] = list(ci_lin["CMC@5_CI"])
        res_lin["CMC@10_CI95"]= list(ci_lin["CMC@10_CI"])
        res_lin["Precision@1_CI95"] = list(ci_lin["Precision@1_CI"])
        res_lin["Precision@5_CI95"] = list(ci_lin["Precision@5_CI"])
        res_lin["Precision@10_CI95"]= list(ci_lin["Precision@10_CI"])
        res_lin["FP@1_CI95"]  = list(ci_lin["FP@1_CI"])
        res_lin["FP@5_CI95"]  = list(ci_lin["FP@5_CI"])
        res_lin["FP@10_CI95"] = list(ci_lin["FP@10_CI"])
        results["linear_sgd"] = res_lin
        tqdm.write("linear_sgd:\n" + json.dumps(res_lin, indent=2))

    # RBF approx
    if CONFIG["RUN_RBF_APPROX"]:
        gamma = (1.0 / CONFIG["PCA_DIM"]) if (CONFIG["RFF_GAMMA"] is None) else CONFIG["RFF_GAMMA"]
        q_rbf, g_rbf = rbf_approx_embeddings(Xtr, ytr, Xq, Xg, CONFIG["RFF_COMPONENTS"], gamma)
        res_rbf, ap_rbf, cmc_rbf, prec_rbf = cmc_map(
            q_rbf[q_idx], q_ref, g_rbf[g_idx], g_ref, gt_map, jk_map
        )
        ci_rbf = bootstrap(np.array(ap_rbf), cmc_rbf, prec_rbf,
                                    n_boot=CONFIG["BOOT_N"], alpha=alpha, seed=CONFIG["SEED"])
        res_rbf["mAP_CI95"]   = list(ci_rbf["mAP_CI"])
        res_rbf["CMC@1_CI95"] = list(ci_rbf["CMC@1_CI"])
        res_rbf["CMC@5_CI95"] = list(ci_rbf["CMC@5_CI"])
        res_rbf["CMC@10_CI95"]= list(ci_rbf["CMC@10_CI"])
        res_rbf["Precision@1_CI95"] = list(ci_rbf["Precision@1_CI"])
        res_rbf["Precision@5_CI95"] = list(ci_rbf["Precision@5_CI"])
        res_rbf["Precision@10_CI95"]= list(ci_rbf["Precision@10_CI"])
        res_rbf["FP@1_CI95"]  = list(ci_rbf["FP@1_CI"])
        res_rbf["FP@5_CI95"]  = list(ci_rbf["FP@5_CI"])
        res_rbf["FP@10_CI95"] = list(ci_rbf["FP@10_CI"])
        results["rbf_approx"] = res_rbf
        tqdm.write("rbf_approx:\n" + json.dumps(res_rbf, indent=2))

    save_json(OUT_FILE, {
        "config": CONFIG,
        "results": results,
        "timestamp": timestamp()
    })
    tqdm.write(f"saved -> {OUT_FILE}")

if __name__ == "__main__":
    main()
