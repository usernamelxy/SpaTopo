import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# ==========================================
# 底层网络组件 (Components)
# ==========================================

class NativeTopologyDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, dropout_rate=0.5):
        super(NativeTopologyDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.d_drop = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, z, edge_indices):
        h = z[edge_indices[0]] * z[edge_indices[1]]
        h = self.fc2(self.activation(self.fc1(self.d_drop(h))))
        return h.squeeze(-1)


def random_negative_sampler_native(num_nodes, num_neg_samples, device):
    return torch.randint(0, num_nodes, size=(2, num_neg_samples), device=device)


def contrastive_ce_loss_hard(pos_out, neg_out_topo, neg_out_feat, task_weights):
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss_topo = F.binary_cross_entropy_with_logits(neg_out_topo, torch.zeros_like(neg_out_topo))
    neg_loss_feat = F.binary_cross_entropy_with_logits(neg_out_feat, torch.zeros_like(neg_out_feat))

    prec_topo = torch.exp(-torch.clamp(task_weights[0], min=-10.0, max=10.0))
    prec_feat = torch.exp(-torch.clamp(task_weights[1], min=-10.0, max=10.0))

    loss_topo_weighted = prec_topo * neg_loss_topo + task_weights[0]
    loss_feat_weighted = prec_feat * neg_loss_feat + task_weights[1]

    return pos_loss + loss_topo_weighted + loss_feat_weighted


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return self.act(output)


# ==========================================
# 核心网络模块 (SpaTopo_module)
# ==========================================

class SpaTopo_module(nn.Module):
    def __init__(
            self, input_dim, feat_hidden1=64, feat_hidden2=16, gcn_hidden1=64, gcn_hidden2=16,
            p_drop=0.2, alpha=1.0, edge_drop_rate=0.3, mask_rate_block=0.4, mask_rate_random=0.8, mask_strategy='block',
            use_mask=True, use_topo=True, use_reparam=True, use_feat_neg=True
    ):
        super(SpaTopo_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2
        self.edge_drop_rate = edge_drop_rate
        self.mask_rate_block = mask_rate_block
        self.mask_rate_random = mask_rate_random
        self.mask_strategy = mask_strategy
        self.use_mask = use_mask
        self.use_topo = use_topo
        self.use_reparam = use_reparam
        self.use_feat_neg = use_feat_neg

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)

        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)

        self.topo_decoder = NativeTopologyDecoder(self.latent_dim, dropout_rate=self.p_drop)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.criterion = partial(sce_loss, alpha=3)
        self.task_weights = nn.Parameter(torch.zeros(2))

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        mu_gcn = self.gc2(hidden1, adj)
        logvar = self.gc3(hidden1, adj)
        return mu_gcn, logvar, feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        adj_coalesced = adj.coalesce()
        edge_indices = adj_coalesced.indices()
        edge_values = adj_coalesced.values()
        num_nodes = x.shape[0]

        ed = self.edge_drop_rate if self.use_topo else 0.0
        p = torch.empty(edge_indices.shape[1], device=adj.device).fill_(1 - ed)
        stay = torch.bernoulli(p).to(torch.bool)
        self_loops = (edge_indices[0] == edge_indices[1])
        stay = stay | self_loops

        remaining_edges = edge_indices[:, stay]
        remaining_values = edge_values[stay]
        masked_edges = edge_indices[:, ~stay]
        masked_adj = torch.sparse_coo_tensor(remaining_edges, remaining_values, adj.shape).coalesce()
        if self.use_mask:
            use_adj, x_masked, (mask_nodes, keep_nodes) = self.encoding_mask_noise(masked_adj, x, self.mask_rate_block,
                                                                                   self.mask_rate_random)
        else:
            use_adj, x_masked = masked_adj, x
            mask_nodes = torch.arange(x.shape[0], device=x.device)

        mu, logvar, feat_x = self.encode(x_masked, use_adj)
        gnn_z = self.reparameterize(mu, logvar) if self.use_reparam else mu
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z, use_adj)

        if self.use_feat_neg:
            rand_idx = torch.randperm(num_nodes, device=x.device)
            x_shuffled = x[rand_idx]
            mu_neg, _, feat_x_neg = self.encode(x_shuffled, use_adj)
            z_neg = torch.cat((feat_x_neg, mu_neg), 1)

        recon = de_feat.clone()
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss_feat = self.criterion(x_rec, x_init)

        num_neg_samples = masked_edges.shape[1]
        if num_neg_samples > 0:
            neg_edges = random_negative_sampler_native(num_nodes, num_neg_samples, z.device)
            pos_out = self.topo_decoder(z, masked_edges)
            neg_out_topo = self.topo_decoder(z, neg_edges)
            if self.use_feat_neg:
                neg_out_feat = self.topo_decoder(z_neg, masked_edges)
                loss_topo = contrastive_ce_loss_hard(pos_out, neg_out_topo, neg_out_feat, self.task_weights)
            else:
                loss_topo = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out)) + \
                            F.binary_cross_entropy_with_logits(neg_out_topo, torch.zeros_like(neg_out_topo))
        else:
            loss_topo = torch.tensor(0.0, device=z.device, requires_grad=True)

        adj_pred = torch.matmul(z, z.t())
        adj_pred = torch.sigmoid(adj_pred)
        adj_dense = use_adj.to_dense()
        loss_topo_recon = F.mse_loss(adj_pred, adj_dense)

        loss_feat_scaled = 10 * loss_feat
        loss_topo_recon_scaled = 0.1 * loss_topo_recon
        if self.use_topo:
            loss = loss_feat_scaled + loss_topo + loss_topo_recon_scaled
        else:
            loss = loss_feat_scaled
        return z, mu, logvar, de_feat, feat_x, gnn_z, loss, loss_feat_scaled, loss_topo, loss_topo_recon_scaled

    def encoding_mask_noise(self, adj, x, mask_rate_block, mask_rate_random):
        if self.mask_strategy == 'block':
            return self._block_masking(adj, x, mask_rate_block)
        else:
            return self._random_masking(adj, x, mask_rate_random)

    def _random_masking(self, adj, x, mask_rate_random):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate_random * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] += self.enc_mask_token
        return adj.clone(), out_x, (mask_nodes, keep_nodes)

    def _block_masking(self, adj, x, mask_rate_block):
        num_nodes = adj.shape[0]
        num_mask_nodes = int(mask_rate_block * num_nodes)
        adj_dense = (adj.to_dense() != 0)
        mask_nodes_set = set()
        candidate_seeds = torch.randperm(num_nodes, device=x.device).tolist()

        for seed in candidate_seeds:
            if len(mask_nodes_set) >= num_mask_nodes:
                break
            if seed in mask_nodes_set:
                continue
            neighbors = torch.nonzero(adj_dense[seed]).squeeze(1).tolist()
            mask_nodes_set.update(neighbors)

        mask_nodes_list = list(mask_nodes_set)
        if len(mask_nodes_list) > num_mask_nodes:
            perm = torch.randperm(len(mask_nodes_list)).tolist()
            mask_nodes_list = [mask_nodes_list[i] for i in perm][:num_mask_nodes]

        mask_nodes = torch.tensor(mask_nodes_list, device=x.device, dtype=torch.long)
        keep_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        keep_mask[mask_nodes] = False
        keep_nodes = torch.nonzero(keep_mask).squeeze(1)

        out_x = x.clone()
        out_x[mask_nodes] += self.enc_mask_token
        return adj.clone(), out_x, (mask_nodes, keep_nodes)


# ==========================================
# 顶层训练器 (SpaTopo Class)
# ==========================================

class SpaTopo:
    def __init__(
            self, X, graph_dict, rec_w=10, self_w=1, device='cuda:0',
            mask_rate_block=0.4, mask_rate_random=0.8, edge_drop_rate=0.3, mask_strategy='block',
            feat_hidden1=64, feat_hidden2=16, gcn_hidden1=64, gcn_hidden2=16, p_drop=0.2,
            use_mask=True, use_topo=True, use_reparam=True, use_feat_neg=True
    ):
        self.rec_w = rec_w
        self.self_w = self_w
        self.device = device
        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.adj_norm = graph_dict["adj_norm"].to(self.device)

        self.model = SpaTopo_module(
            self.input_dim, mask_rate_block=mask_rate_block, mask_rate_random=mask_rate_random,
            edge_drop_rate=edge_drop_rate, mask_strategy=mask_strategy,
            feat_hidden1=feat_hidden1, feat_hidden2=feat_hidden2,
            gcn_hidden1=gcn_hidden1, gcn_hidden2=gcn_hidden2, p_drop=p_drop,
            use_mask=use_mask, use_topo=use_topo, use_reparam=use_reparam,
            use_feat_neg=use_feat_neg
        ).to(self.device)

    def process(self):
        self.model.eval()
        with torch.no_grad():
            latent_z, _, _, _, feat_x, gnn_z, _, _, _, _ = self.model(self.X, self.adj_norm)
        return latent_z.data.cpu().numpy(), feat_x.data.cpu().numpy(), gnn_z.data.cpu().numpy()

    def recon(self):
        self.model.eval()
        with torch.no_grad():
            _, _, _, de_feat, _, _, _, _, _, _ = self.model(self.X, self.adj_norm)
        return StandardScaler().fit_transform(de_feat.data.cpu().numpy())

    def train(self, epochs=200, lr=0.01, decay=0.01):
        optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=decay)
        print("开始自监督表征训练 (Pure Self-Supervised)...")
        print(f"{'Epoch':<8} {'Total':<10} {'Feat_SCE':<10} {'Topo_CL':<10} {'Topo_MSE':<10}")
        print("-" * 50)
        loss_history = []

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            _, _, _, _, _, _, _, loss_feat, loss_topo, loss_topo_recon = self.model(self.X, self.adj_norm)
            loss = self.self_w * (loss_feat + loss_topo + loss_topo_recon)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if (epoch + 1) % 20 == 0:
                print(f"{epoch+1:<8} {loss.item():<10.4f} {loss_feat.item():<10.4f} {loss_topo.item():<10.4f} {loss_topo_recon.item():<10.4f}")
                if (epoch + 1) % 20 == 0:
                    pbar.set_description(f"Epoch {epoch+1}")

        print("-" * 50)
        print(f"Final: Total={loss_history[-1]:.4f}")
        return loss_history