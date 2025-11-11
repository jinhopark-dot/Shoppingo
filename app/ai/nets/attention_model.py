import torch
from torch import nn
from typing import NamedTuple
import math

from .great import GREATEncoder
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ✅ set_decode_type 함수를 이 파일에 다시 정의합니다.
def set_decode_type(model, decode_type, temp=None):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type, temp)

class PointerformerFixed(NamedTuple):
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    end_node_emb: torch.Tensor

class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=5, n_heads=8, latent_dim=16, **kwargs):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.decode_type = None
        self.temp = 1.0
        self.problem = problem
        self.n_heads = n_heads
        self.latent_dim = latent_dim

        self.encoder = GREATEncoder(
            initial_dim=1, hidden_dim=embedding_dim, num_layers=n_encode_layers,
            num_nodes=problem.size, heads=n_heads, final_node_layer=True
        )
        
        context_input_dim = embedding_dim + latent_dim
        
        self.Wq_graph = nn.Linear(context_input_dim, embedding_dim, bias=False)
        self.Wq_end = nn.Linear(context_input_dim, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(context_input_dim, embedding_dim, bias=False)
        self.W_visited = nn.Linear(context_input_dim, embedding_dim, bias=False)
        
        self.Wk = nn.Linear(embedding_dim + latent_dim, embedding_dim, bias=False)
        
        self.tanh_clipping = kwargs.get('tanh_clipping', 10.)

    def forward(self, input, z, return_pi=False):
        embeddings_flat = self.encoder(input) 
        embeddings, _ = to_dense_batch(embeddings_flat, input.batch)
        dist_matrix_batch = to_dense_adj(input.edge_index, input.batch, input.edge_attr.squeeze(-1))
        
        decoder_input = {
            'dist_matrix': dist_matrix_batch,
            'start_idx': input.start_idx,
            'end_idx': input.end_idx
        }
        
        _log_p, pi = self._inner(decoder_input, embeddings, z)
        
        cost, mask = self.problem.get_costs(decoder_input, pi)
        ll = self._calc_log_likelihood(_log_p, pi[:, 1:], mask)
        
        if return_pi:
            return cost, ll, pi
        return cost, ll
    
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type, self.temp = decode_type, temp if temp is not None else self.temp
    
    def _precompute(self, embeddings, state):
        graph_embedding = embeddings.mean(dim=1, keepdim=True)
        end_node_emb = embeddings.gather(1, state.get_end_node().unsqueeze(-1).expand(-1, -1, self.embedding_dim))
        
        return PointerformerFixed(embeddings, graph_embedding, end_node_emb)
        
    def _inner(self, input, embeddings, z):
        outputs, sequences = [], []
        state = self.problem.make_state(input)
        
        fixed = self._precompute(embeddings, state)

        sequences.append(state.get_current_node().squeeze(-1))
        while not state.all_finished():
            log_p, mask = self._get_log_p(fixed, state, z)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            state = state.update(selected)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _get_log_p(self, fixed, state, z):
        B, N, D = fixed.node_embeddings.shape
        H = self.n_heads
        D_h = D // H
        
        z_expanded_query = z.unsqueeze(1)
        z_expanded_key = z.unsqueeze(1).expand(B, N, self.latent_dim)
        
        embeddings_conditioned = torch.cat([fixed.node_embeddings, z_expanded_key], dim=-1)
        k = self.Wk(embeddings_conditioned).view(B, N, H, D_h).transpose(1, 2)
        
        q_graph = self.Wq_graph(torch.cat([fixed.graph_embedding, z_expanded_query], dim=-1))
        q_end = self.Wq_end(torch.cat([fixed.end_node_emb, z_expanded_query], dim=-1))
        
        h_current = fixed.node_embeddings.gather(1, state.get_current_node().unsqueeze(-1).expand(-1, -1, D))
        q_last = self.Wq_last(torch.cat([h_current, z_expanded_query], dim=-1))
        
        tour_nodes = state.tour_
        tour_emb = fixed.node_embeddings.gather(1, tour_nodes.unsqueeze(-1).expand(-1, -1, D))
        h_tour = tour_emb.sum(dim=1, keepdim=True) / tour_nodes.size(1)
        q_visited = self.W_visited(torch.cat([h_tour, z_expanded_query], dim=-1))
        
        final_q = q_graph + q_end + q_last + q_visited
        q = final_q.view(B, 1, H, D_h).transpose(1, 2)
        
        compatibility = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_h)
        avg_compat = compatibility.squeeze(2).mean(dim=1).unsqueeze(1)
        
        current_node_idx = state.get_current_node().squeeze(-1)
        batch_idx = torch.arange(B, device=current_node_idx.device)
        distances = state.dist_matrix[batch_idx, current_node_idx].unsqueeze(1)
        
        logits = avg_compat - (distances / math.sqrt(D))
        
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        
        mask = state.get_mask()
        logits[mask] = -math.inf

        log_p = torch.log_softmax(logits / self.temp, dim=-1)
            
        return log_p, mask
    
    def _select_node(self, probs, mask):
        assert (probs[mask] == 0).all()
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            log_p[mask] = 0
        return log_p.sum(1)