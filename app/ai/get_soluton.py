import os
import json
import torch
import argparse
import pickle
import numpy as np
import pprint as pp
import torch.nn.functional as F

from torch_geometric.data import Batch, Data

from .problems.shopping import Shopping
from .nets.attention_model import AttentionModel, set_decode_type
from .train import get_inner_model
from .utils import torch_load_cpu, move_to

def _create_subgraph_from_labels(full_data, node_labels_to_extract, start_node_label, device):
    """
    [내부 헬퍼 함수]
    전체 그래프 데이터에서 사용자가 요청한 노드들만 추출하고,
    지정된 시작 노드를 설정하여 새로운 PyG Data 객체를 생성합니다.
    """
    print(f"\n  [INFO] {len(node_labels_to_extract)}개의 노드를 추출하여 서브그래프 생성 중...")
    
    label_to_idx = full_data['label_to_idx']
    try:
        indices = [label_to_idx[label] for label in node_labels_to_extract]
    except KeyError as e:
        print(f"  [오류] 레이블 '{e.args[0]}'를 전체 그래프('full_data')에서 찾을 수 없습니다.")
        return None

    N_sub = len(indices)
    
    sub_dist_mat = full_data['dist_matrix'][np.ix_(indices, indices)]
    sub_coords = full_data['node_coords'][indices]

    try:
        end_label = full_data['labels'][full_data['default_end_idx']]
        sub_end_idx = node_labels_to_extract.index(end_label)
        
        if start_node_label not in node_labels_to_extract:
            raise ValueError(f"지정한 시작 노드 '{start_node_label}'가 쇼핑 리스트에 포함되어 있지 않습니다.")
            
        sub_start_idx = node_labels_to_extract.index(start_node_label)
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"  [오류] 시작/종료 노드 인덱스 생성 중 오류: {e}")
        return None

    dist_mat_torch = torch.FloatTensor(sub_dist_mat).to(device)
    
    adj = torch.ones((N_sub, N_sub)) - torch.eye(N_sub)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(device)
    edge_attr = dist_mat_torch[edge_index[0], edge_index[1]].unsqueeze(1)
    
    data = Data(
        edge_index=edge_index, edge_attr=edge_attr, dist_matrix=dist_mat_torch,
        node_coords=torch.FloatTensor(sub_coords).to(device),
        start_idx=torch.tensor(sub_start_idx, dtype=torch.long).to(device),
        end_idx=torch.tensor(sub_end_idx, dtype=torch.long).to(device),
        num_nodes=N_sub
    )
    return data


# ⭐️ 1. [신규 함수] 서버 시작 시 1회만 실행될 함수
def load_ai_assets(load_path, full_graph_path, use_cuda=True):
    """
    서버 시작 시 AI 모델(.pt), 설정(args.json), 데이터(.pkl)를
    미리 로드하여 메모리에 올립니다.
    """
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"  [INFO] AI 자산을 {device} 디바이스로 로드합니다.")

    # --- 1. 훈련 당시의 설정(opts) 불러오기 ---
    load_dir = os.path.dirname(load_path)
    args_json_path = os.path.join(load_dir, "args.json")
    
    if not os.path.exists(args_json_path):
        print(f"  [오류] {args_json_path} 에서 args.json을 찾을 수 없습니다.")
        raise FileNotFoundError(f"args.json not found at {args_json_path}")
        
    print(f"  [INFO] {args_json_path} 에서 훈련 설정을 로드합니다.")
    with open(args_json_path, 'r') as f:
        train_opts_dict = json.load(f)
    
    opts = argparse.Namespace(**train_opts_dict)
    opts.device = device

    # --- 2. 모델 로드 (단 한번) ---
    print(f"  [INFO] {load_path} 에서 모델 로드 중...")
    problem = Shopping()
    problem.size = opts.graph_size 
    
    model = AttentionModel(
        embedding_dim=opts.embedding_dim, hidden_dim=opts.hidden_dim,
        problem=problem, n_encode_layers=opts.n_encode_layers,
        n_heads=opts.n_heads, tanh_clipping=opts.tanh_clipping,
        latent_dim=opts.latent_dim
    ).to(opts.device)
    
    load_data = torch_load_cpu(load_path)
    get_inner_model(model).load_state_dict(load_data['model'])
    model.eval()
    print("  [INFO] 모델 로드 완료.")

    # --- 3. 전체 그래프 데이터 로드 (단 한번) ---
    print(f"  [INFO] {full_graph_path} 에서 전체 그래프 데이터 로드 중...")
    try:
        with open(full_graph_path, 'rb') as f:
            full_data = pickle.load(f)
    except Exception as e:
        print(f"  [오류] 전체 그래프 파일 로드 실패: {e}")
        raise FileNotFoundError(f"Failed to load {full_graph_path}")
    print("  [INFO] 전체 그래프 로드 완료.")
    
    #  3가지를 반환: (모델 객체, 설정 객체, 데이터 객체)
    return model, opts, full_data


#  2. API 호출 시 매번 실행될 가벼운 추론 함수
def run_ai_inference(model, opts, full_data, shopping_list, start_node_label, num_samples):
    """
    미리 로드된 AI 자산(model, opts, full_data)을 받아
    '추론'만 수행합니다. (I/O 작업 없음)
    """
    
    # --- 4. 입력된 정보로 PyG 인스턴스(서브그래프) 생성 ---
    instance = _create_subgraph_from_labels(
        full_data, 
        shopping_list, 
        start_node_label,
        opts.device
    )
    if instance is None:
        print("  [오류] 서브그래프 인스턴스 생성 실패.")
        return None #  None 반환

    if instance.num_nodes != opts.graph_size:
        get_inner_model(model).problem.size = instance.num_nodes

    # --- 5. MP-ASIL 샘플링으로 최고의 해 탐색 ---
    print(f"  [INFO] {num_samples}번의 샘플링 수행")
    set_decode_type(get_inner_model(model), "sampling", temp=1.0)
    
    B = 1; k = num_samples
    num_continuous = 4; num_categorical = 12
    assert opts.latent_dim == num_continuous + num_categorical
    
    z_continuous = torch.rand(B, k, num_continuous, device=opts.device) * 2 - 1
    indices = torch.randint(0, num_categorical, (B, k), device=opts.device)
    z_categorical = F.one_hot(indices, num_classes=num_categorical).float()
    z = torch.cat([z_continuous, z_categorical], dim=-1)

    x_expanded = Batch.from_data_list([instance for _ in range(k)])
    z_reshaped = z.view(B * k, opts.latent_dim)
    
    with torch.no_grad():
        costs, _, pis = model(x_expanded, z_reshaped, return_pi=True)
    
    best_idx = torch.argmin(costs)
    best_cost = costs[best_idx].item()
    best_pi_indices = pis[best_idx].cpu().numpy()
    
    # --- 6. 최종 결과를 인덱스에서 다시 레이블로 변환 ---
    final_path_labels = [shopping_list[i] for i in best_pi_indices]
    
    print(f"  [INFO] 추론 완료. 최적 비용: {best_cost:.4f}")
    
    return final_path_labels, best_cost