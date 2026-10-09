import scanpy as sc
import pandas as pd
import numpy as np

def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=2023):

    for res in np.arange(0.2, 2, increment):
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['leiden'].unique()) > n_clusters:
            break
    return res-increment


def leiden(adata, n_clusters, key_added='SpaTopo',use_rep='SpaTopo', random_seed=42):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_leiden(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.leiden(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['leiden']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=2023):
    for res in np.arange(0.2, 2, increment):
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        if len(adata.obs['louvain'].unique()) > n_clusters:
            break
    return res-increment

def louvain(adata, n_clusters, key_added='SpaTopo',use_rep='SpaTopo', random_seed=42):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res = res_search_fixed_clus_louvain(adata, n_clusters, increment=0.01, random_seed=random_seed)
    sc.tl.louvain(adata, random_state=random_seed, resolution=res)

    adata.obs[key_added] = adata.obs['louvain']
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata

def mclust_R(adata, n_clusters, use_rep='SpaTopo', key_added='SpaTopo'):
    """
    Revised version: Run R script directly to ensure data is treated as a matrix.
    This bypasses rpy2 function call interface issues.
    """
    # --- 1. 必须在这里导入所有需要的库 ---
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import conversion  # <--- 你之前的报错就是因为缺了这一行

    # --- 2. 检查 R 包 ---
    try:
        robjects.r.library("mclust")
    except Exception:
        raise RuntimeError("R package 'mclust' is not installed. Please install it in R.")

    # --- 3. 数据准备 ---
    data_np = adata.obsm[use_rep]
    # 强制连续内存和 float64
    data_np = np.ascontiguousarray(data_np, dtype=np.float64)

    # --- 4. 数据传输 (Python -> R) ---
    # 使用 conversion 模块将 numpy 数组放入 R 环境
    with conversion.localconverter(robjects.default_converter + numpy2ri.converter):
        robjects.globalenv['r_data'] = data_np

    # --- 5. R 脚本执行 ---
    r_script = f'''
    library(mclust)
    set.seed(42)
    # 强制转换为矩阵，修复维度丢失问题
    r_data <- as.matrix(r_data)

    # 运行聚类
    res <- Mclust(r_data, G={n_clusters}, modelNames="EEE")

    # 返回分类结果
    res$classification
    '''

    print(f"Running Mclust in R with {n_clusters} clusters...")

    try:
        res = robjects.r(r_script)
    except Exception as e:
        print("R execution error:")
        print(e)
        raise

    # --- 6. 结果处理 ---
    mclust_res = np.array(res)
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata

