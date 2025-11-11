import torch
import os
import json


def load_problem(name):
    """
    Load problem class by name
    """
    from problems.shopping.problem_shopping import Shopping
    
    if name == 'shopping':
        return Shopping
    else:
        raise ValueError(f"Unknown problem: {name}. Only 'shopping' is supported.")


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def load_model(path, epoch=None):
    """Load trained model"""
    # from nets.attention_model import AttentionModel # 이 import는 맨 위로 이동했습니다.

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, f'epoch-{epoch}.pt')
    else:
        raise ValueError(f"{path} is not a valid directory or file")

    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])
    
    # 🔑 problem 객체에 graph_size를 명시적으로 저장
    problem.size = args.get('graph_size')
    
    from nets.attention_model import AttentionModel
    # 🔑 AttentionModel 호출 시, 새로운 GREAT 아키텍처에 맞는 인자로 변경
    model = AttentionModel(
        embedding_dim=args['embedding_dim'],
        hidden_dim=args['hidden_dim'],
        problem=problem,
        n_encode_layers=args.get('n_encode_layers'),
        normalization=args.get('normalization'),
        n_heads=args.get('n_heads'),
        tanh_clipping=args.get('tanh_clipping'),
        nodeless=args.get('nodeless', False),
        dropout=args.get('dropout', 0.1)
    )

    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model.eval()

    return model, args


# ❌ 삭제: _load_model_file() 함수
# ❌ 삭제: parse_softmax_temperature() 함수
# ❌ 삭제: run_all_in_pool() 함수
# ❌ 삭제: do_batch_rep() 함수
# ❌ 삭제: sample_many() 함수