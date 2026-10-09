

import os
import argparse
import warnings
from pathlib import Path
import scanpy as sc
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import gc
from sklearn.decomposition import PCA
from SpaTopo.utils_func import  fix_seed, calculate_moran_i_routing
from SpaTopo.graph_func import graph_construction
from SpaTopo.model import SpaTopo
from SpaTopo.clustering_func import mclust_R

import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpaTopo for Spatial Transcriptomics")
    parser.add_argument('--data_root', type=str, default='./data/DLPFC', help='Root directory of data')
    parser.add_argument('--sample_names', type=str, nargs='+', default=['151507', '151508','151509', '151510', '151669', '151670','151671','151672', '151673','151674', '151675','151676'],
                        help='List of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Preprocessing
    parser.add_argument('--min_cells', type=int, default=50)
    parser.add_argument('--min_counts', type=int, default=10)
    parser.add_argument('--target_sum', type=float, default=1e6)
    parser.add_argument('--n_top_genes', type=int, default=2000)
    parser.add_argument('--pca_n_comps', type=int, default=200)

    # Graph & Routing
    parser.add_argument('--n_neighbors', type=int, default=12)
    parser.add_argument('--mask_rate_block', type=float, default=0.4)
    parser.add_argument('--mask_rate_random', type=float, default=0.8)
    parser.add_argument('--edge_drop_rate', type=float, default=0.3)

    # Model
    parser.add_argument('--feat_hidden1', type=int, default=64)
    parser.add_argument('--feat_hidden2', type=int, default=16)
    parser.add_argument('--gcn_hidden1', type=int, default=64)
    parser.add_argument('--gcn_hidden2', type=int, default=16)
    parser.add_argument('--p_drop', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    return parser.parse_args()


def process_single_sample(args, sample_name):
    print("\n" + "=" * 50)
    print(f"🚀 Starting Processing Dataset: {sample_name}")
    print("=" * 50)

    fix_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_root = Path(args.data_root)
    num_clust = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7

    print("Loading data...")
    adata = sc.read_visium(data_root / sample_name)
    adata.var_names_make_unique()

    df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    adata.obs['layer_guess'] = df_meta['layer_guess']

    print("Preprocessing...")
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    sc.pp.filter_genes(adata, min_counts=args.min_counts)
    sc.pp.normalize_total(adata, target_sum=args.target_sum)

    # 1. 筛选高变异基因 (此时 adata.X 已经被 normalize，但没有被 scale，全是正数)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=args.n_top_genes)
    adata = adata[:, adata.var['highly_variable'] == True]

    # ==========================================
    # 步骤 A: 计算路由策略
    # 放在 scale 之前，确保用纯净的高变异基因计算余弦距离
    # ==========================================
    # print("Calculating routing strategy...")
    # mask_strategy = calculate_moran_i_routing(
    #     adata, n_features=500, n_neighbors=12, threshold=0.10
    # )
    # ==========================================
    # 步骤 A: 计算路由策略
    # ==========================================
    print("Calculating routing strategy...")
    mask_strategy = calculate_moran_i_routing(
        adata,
        n_hvg_candidates=1000,  # 替代原来的 n_features，先挑1000个高变基因
        n_svgs=50,  # 选出 top 50 的空间高变基因算 Moran's I
        n_neighbors=12,
        threshold=0.30  # 阈值同步调高
    )
    # ==========================================
    # 步骤 B: 构建空间邻接图
    # 仅依赖 adata.obsm['spatial']，放在这里完全不受基因操作的影响
    # ==========================================
    print("Constructing graph...")
    graph_dict = graph_construction(adata, n=args.n_neighbors)

    # ==========================================
    # 步骤 C: Scale 与 PCA (专为 GNN 模型准备输入特征)
    # 这一步会破坏 adata.X，但这不要紧，因为路由策略已经算完保存好了
    # ==========================================
    print("Scaling and PCA for model input...")
    sc.pp.scale(adata)
    adata_X = PCA(n_components=args.pca_n_comps, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    # ==========================================
    # 步骤 D: 训练模型
    # ==========================================
    print("Training SpaTopo...")
    model = SpaTopo(
        adata.obsm['X_pca'], graph_dict, device=device,
        mask_rate_block=args.mask_rate_block, mask_rate_random=args.mask_rate_random,
        edge_drop_rate=args.edge_drop_rate, mask_strategy=mask_strategy,
        feat_hidden1=args.feat_hidden1, feat_hidden2=args.feat_hidden2,
        gcn_hidden1=args.gcn_hidden1, gcn_hidden2=args.gcn_hidden2, p_drop=args.p_drop
    )

    loss_history = model.train(epochs=args.epochs, lr=args.lr, decay=args.weight_decay)

    # 提取特征
    sedr_feat, feat_x, gnn_z = model.process()
    adata.obsm['SpaTopo'] = sedr_feat
    adata.obsm['spatopo_recon_pca'] = model.recon()

    print("Clustering...")
    mclust_R(adata, num_clust, use_rep='SpaTopo', key_added='SpaTopo')

    # 评估与绘图
    sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
    ARI = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['SpaTopo'])
    print(f"\n>>> Final ARI for {sample_name}: {ARI:.4f} <<<\n")

    save_dir = f'figures/recon/SpaTopo/{sample_name}'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), loss_history, label='Loss', color='#FF6B6B', linewidth=2)
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.title(f'Loss Curve - {sample_name}'), plt.legend(), plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, f'loss_curve_{sample_name}.png'), dpi=300), plt.close()

    adata.write_h5ad(os.path.join(save_dir, 'SpaTopo.h5ad'))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sc.pl.spatial(adata, color='layer_guess', ax=axes[0], show=False)
    sc.pl.spatial(adata, color='SpaTopo', ax=axes[1], show=False)
    axes[0].set_title('Manual Annotation'), axes[1].set_title(f'Clustering: (ARI={ARI:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spatial_domain_{sample_name}.png'), dpi=300), plt.close()

    print(f"✅ Pipeline completed for {sample_name}.\n")

    del adata, model, graph_dict, loss_history
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    args = parse_args()
    for sample in args.sample_names:
        try:
            process_single_sample(args, sample)
        except Exception as e:
            print(f"❌ Error processing sample {sample}: {e}")
            continue


if __name__ == "__main__":
    main()
