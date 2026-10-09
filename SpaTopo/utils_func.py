import os
import random
import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import pairwise_distances


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    from torch.backends import cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def adata_preprocess(adata, min_cells=50, min_counts=10, target_sum=1e6, n_top_genes=2000, pca_n_comps=200):
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable'] == True]

    sc.pp.scale(adata)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_n_comps, random_state=42)
    adata_X = pca.fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    adata.uns['pca_obj'] = pca
    return adata


import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

#
# def calculate_moran_i_routing(adata, n_features=500, n_neighbors=12, threshold=0.10, return_value=False):
#     """
#     用 Moran's I（空间自相关）替代 Dirichlet 能量做掩码路由决策。
#
#     原理:
#       - 对每个 HVG 计算 Moran's I
#       - 取所有基因的均值 mean_I
#       - mean_I 高 → 邻近 spots 表达相似 → 空间结构清晰 → RANDOM 足够
#       - mean_I 低 → 邻近 spots 表达随机 → 空间结构复杂 → 需要 BLOCK
#
#     参数:
#         adata: AnnData，已 normalize（log1p）但未 scale
#         n_features: 使用前 N 个 HVG
#         n_neighbors: 空间邻居数（k-NN）
#         threshold: Moran's I 阈值
#         return_value: 若 True，返回 (mask_strategy, mean_I) 元组
#                    mean_I >= threshold → 'random'
#                    mean_I <  threshold → 'block'
#
#     返回:
#         mask_strategy: 'block' 或 'random'
#     """
#     import numpy as np
#     from sklearn.metrics import pairwise_distances
#
#     # --- 1. 特征选择（与 calculate_dirichlet_energy_routing 一致的 HVG 逻辑）---
#     if 'variances_norm' in adata.var.columns:
#         top_idx = adata.var['variances_norm'].argsort()[::-1][:n_features]
#         features = adata.X[:, top_idx]
#     elif 'dispersions_norm' in adata.var.columns:
#         top_idx = adata.var['dispersions_norm'].argsort()[::-1][:n_features]
#         features = adata.X[:, top_idx]
#     else:
#         # 兜底：没有 HVG 指标时，用 sklearn PCA 降维代替
#         # 注意：不用 sc.tl.pca（会写入 adata.obsm['X_pca']，破坏下游模型输入）
#         print("Warning: HVG metrics not found. Falling back to sklearn PCA for Moran's I routing...")
#         from sklearn.decomposition import PCA
#         from sklearn.preprocessing import StandardScaler
#         if hasattr(adata.X, "toarray"):
#             X_dense = adata.X.toarray()
#         else:
#             X_dense = adata.X.copy()
#         X_scaled = StandardScaler().fit_transform(X_dense)
#         n_pcs = min(n_features, 50, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1)
#         pca = PCA(n_components=n_pcs, random_state=42)
#         features = pca.fit_transform(X_scaled)
#         n_features = features.shape[1]  # 用实际 PCA 维度覆盖原来的 n_features
#         print(f"  Using {n_features} PCs as feature proxies (no side effects on adata).")
#
#     if hasattr(features, "toarray"):
#         features = features.toarray()
#
#     # --- 2. 空间 KNN 图 ---
#     coords = adata.obsm['spatial']
#     dist_mat = pairwise_distances(coords, metric='euclidean')
#     np.fill_diagonal(dist_mat, np.inf)
#     k_neighbors = np.argsort(dist_mat, axis=1)[:, :n_neighbors]
#
#     # --- 3. 逐基因计算 Moran's I ---
#     #     I = (N / W) * [ Σ_i Σ_j w_ij (x_i - x̄)(x_j - x̄) ] / [ Σ_i (x_i - x̄)² ]
#     N = features.shape[0]
#     k = n_neighbors
#     W = N * k                     # 每个 spot 恰好有 k 个邻居，总权重 N*k
#
#     moran_values = []
#     for g in range(n_features):
#         x = features[:, g]
#         x_mean = np.mean(x)
#         x_centered = x - x_mean
#         denom = np.sum(x_centered ** 2)
#
#         if denom == 0:
#             moran_values.append(0.0)
#             continue
#
#         # 分子：对所有 i，邻居 j∈N(i) 的 w_ij (x_i - x̄)(x_j - x̄)
#         numer = 0.0
#         for i in range(N):
#             neighbor_vals = x_centered[k_neighbors[i]]
#             numer += x_centered[i] * np.sum(neighbor_vals)
#
#         I_g = (N / W) * (numer / denom)
#         moran_values.append(I_g)
#
#     mean_I = float(np.mean(moran_values))
#     median_I = float(np.median(moran_values))
#
#     # --- 4. 路由决策 ---
#     # 方向与 MCS 相反：
#     #   MCS 高 (≥1.0) → BLOCK（复杂）
#     #   Moran I 高 (≥ threshold) → RANDOM（规律、简单）
#     mask_strategy = 'random' if mean_I >= threshold else 'block'
#
#     print(f"Mean Moran's I (top {n_features} HVGs): {mean_I:.4f} "
#           f"(median={median_I:.4f}) -> Locked: {mask_strategy.upper()}")
#
#     if return_value:
#         return mask_strategy, mean_I
#     return mask_strategy

import numpy as np
from sklearn.metrics import pairwise_distances


def calculate_moran_i_routing(adata, n_hvg_candidates=1000, n_svgs=50, n_neighbors=12, threshold=0.30,
                              return_value=False):
    """
    使用空间高变基因 (SVGs) 的 Moran's I 决定掩码路由。

    参数:
        adata: AnnData，已 normalize
        n_hvg_candidates: 第一步粗筛的候选基因数（根据方差）
        n_svgs: 第二步精筛的真正空间高变基因数（根据 Moran's I）
        n_neighbors: 空间邻居数
        threshold: 决策阈值（注意：由于提取的是 Top SVGs，阈值需比随机基因高得多，推荐 0.3 - 0.4）
    """
    # --- 1. 粗筛：按方差选出候选基因，大幅减少后续计算量 ---
    if 'variances_norm' in adata.var.columns:
        top_idx = adata.var['variances_norm'].argsort()[::-1][:n_hvg_candidates]
    elif 'dispersions_norm' in adata.var.columns:
        top_idx = adata.var['dispersions_norm'].argsort()[::-1][:n_hvg_candidates]
    else:
        # 兜底：直接计算方差
        X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        var_array = np.var(X_data, axis=0)
        top_idx = np.argsort(var_array)[::-1][:n_hvg_candidates]

    features = adata.X[:, top_idx]
    if hasattr(features, "toarray"):
        features = features.toarray()

    N, n_features = features.shape

    # --- 2. 构建空间 KNN 图 ---
    coords = adata.obsm['spatial']
    dist_mat = pairwise_distances(coords, metric='euclidean')
    np.fill_diagonal(dist_mat, np.inf)
    k_neighbors = np.argsort(dist_mat, axis=1)[:, :n_neighbors]
    W = N * n_neighbors

    # --- 3. 计算所有候选基因的 Moran's I ---
    moran_values = []
    for g in range(n_features):
        x = features[:, g]
        x_mean = np.mean(x)
        x_centered = x - x_mean
        denom = np.sum(x_centered ** 2)

        # 过滤掉几乎无表达差异的极值基因
        if denom < 1e-6:
            moran_values.append(0.0)
            continue

        # 向量化计算分子，提升效率
        # 提取所有节点的邻居值并求和，shape: (N,)
        neighbor_vals_sum = np.sum(x_centered[k_neighbors], axis=1)
        numer = np.sum(x_centered * neighbor_vals_sum)

        I_g = (N / W) * (numer / denom)
        moran_values.append(I_g)

    moran_values = np.array(moran_values)

    # --- 4. 精筛：提取真正的 Top SVGs ---
    # 清理可能出现的 NaN
    valid_idx = ~np.isnan(moran_values)
    moran_values = moran_values[valid_idx]

    actual_n_svgs = min(n_svgs, len(moran_values))
    # 降序排列并取出前 n_svgs 个最大的 Moran's I
    top_svg_moran = np.sort(moran_values)[::-1][:actual_n_svgs]

    mean_I = float(np.mean(top_svg_moran))
    median_I = float(np.median(top_svg_moran))

    # --- 5. 路由决策 (已切换为中位数评估) ---
    # median_I 高 (结构清晰) -> RANDOM 掩码
    # median_I 低 (噪音大/结构复杂) -> BLOCK 掩码
    mask_strategy = 'random' if median_I >= threshold else 'block'

    print(f"Moran's I (top {actual_n_svgs} SVGs) - Median: {median_I:.4f} "
          f"(Mean: {mean_I:.4f}) -> Locked: {mask_strategy.upper()}")

    if return_value:
        return mask_strategy, median_I  # 注意这里也改成了返回中位数
    return mask_strategy