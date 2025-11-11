from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from .state_shopping import StateShopping
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data

def generate_pyg_instance(graph_size, edge_index, noise_factor=5, noise_probability=0.3):
    """
    사전 계산된 edge_index를 사용하여 PyG Data 객체를 빠르게 생성합니다.
    """
    # 1단계 ~ 5단계: 이 부분은 이전과 동일하게 매우 빠릅니다.
    node_coords = np.random.uniform(0, 50, size=(graph_size, 2))
    euclidean_dist_matrix = squareform(pdist(node_coords, 'euclidean'))
    upper_triangle_indices = np.triu_indices(graph_size, k=1)
    num_upper_triangle_edges = len(upper_triangle_indices[0])
    num_noisy_edges = int(num_upper_triangle_edges * noise_probability)
    potential_noise = 1 + np.random.uniform(0, noise_factor, size=num_upper_triangle_edges)
    indices_to_apply_noise = np.random.choice(num_upper_triangle_edges, num_noisy_edges, replace=False)
    noise_multipliers = np.ones(num_upper_triangle_edges)
    noise_multipliers[indices_to_apply_noise] = potential_noise[indices_to_apply_noise]
    noise_matrix = np.ones((graph_size, graph_size))
    noise_matrix[upper_triangle_indices] = noise_multipliers
    noise_matrix.T[upper_triangle_indices] = noise_multipliers
    dist_matrix = torch.FloatTensor(euclidean_dist_matrix * noise_matrix)

    start_idx, end_idx = np.random.choice(graph_size, 2, replace=False)

    # --- 🔑 6단계: 더 이상 edge_index를 생성하지 않고, 바로 사용 ---
    edge_attr = dist_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

    return Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_coords=torch.FloatTensor(node_coords),
        dist_matrix=dist_matrix,
        start_idx=torch.tensor(start_idx, dtype=torch.long),
        end_idx=torch.tensor(end_idx, dtype=torch.long),
        num_nodes=graph_size
    )


class Shopping(object):
    NAME = 'shopping'
    
    # ==============================================================================
    #      최종적으로 수정된 get_costs 함수 (벡터화 + 논리 오류 수정)
    # ==============================================================================
    @staticmethod
    def get_costs(dataset, pi):
        """
        투어 비용 계산 (벡터화된 최종 버전)
        pi는 모델이 출력한 전체 경로입니다. (시작 노드 포함)
        """
        dist_matrix = dataset['dist_matrix']
        batch_size, graph_size = pi.size()

        # 1. 경로 비용 계산 (pi[0]->pi[1]->...->pi[n-1])
        batch_idx = torch.arange(batch_size, device=pi.device).unsqueeze(1)
        from_nodes = pi[:, :-1]
        to_nodes = pi[:, 1:]
        
        path_costs = dist_matrix[batch_idx.expand_as(from_nodes), from_nodes, to_nodes].sum(dim=1)

        # 2. 순회 문제(start==end)인 경우, 마지막 노드에서 시작 노드로 돌아오는 비용 추가
        start_idx = dataset['start_idx']
        end_idx = dataset['end_idx']
        
        # 순회 문제에 해당하는 인스턴스의 마스크 생성
        is_tour_mask = (start_idx == end_idx)
        
        if is_tour_mask.any():
            # 순회 문제인 인스턴스에 대해서만 마지막 노드와 시작 노드를 가져옴
            last_nodes = pi[is_tour_mask, -1]
            start_nodes_of_tour = start_idx[is_tour_mask]
            batch_idx_of_tour = torch.arange(is_tour_mask.sum(), device=pi.device)
            
            # 돌아오는 비용 계산
            return_costs = dist_matrix[is_tour_mask, last_nodes, start_nodes_of_tour]
            
            # 전체 비용에 돌아오는 비용을 더해줌
            path_costs[is_tour_mask] += return_costs

        return path_costs, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return ShoppingDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StateShopping.initialize(*args, **kwargs)

class ShoppingDataset(Dataset):
    def __init__(self, size=30, num_samples=10000, **kwargs):
        super(ShoppingDataset, self).__init__()
        
        print(f"Generating {num_samples} PyG shopping instances...")

        # --- 🔑 1. edge_index를 단 한 번만 미리 계산합니다 ---
        adj = torch.ones((size, size)) - torch.eye(size)
        self.edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        
        self.data = []
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples} instances")
            
            # --- 🔑 2. 사전 계산된 edge_index를 인자로 전달합니다 ---
            self.data.append(
                generate_pyg_instance(
                    graph_size=size,
                    edge_index=self.edge_index,
                    **kwargs
                )
            )
            
        self.size = len(self.data)
        print("Dataset generation complete!")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

