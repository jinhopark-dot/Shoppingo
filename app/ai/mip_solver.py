import gurobipy as gp
from gurobipy import GRB
import torch # _create_subgraph_from_labels에서 반환하는 객체 처리를 위해

from .get_soluton import _create_subgraph_from_labels 

# 서브투어 제거(Subtour Elimination) 콜백

def subtourelim(model, where):
    """
    DFJ(Dantzig-Fulkerson-Johnson) 공식을 위한 Lazy Constraint 콜백 함수.
    최적화 중간에 발견되는 해에서 '서브투어(subtour)'(끊어진 경로)가
    있는지 확인하고, 있다면 이를 제거하는 제약조건을 동적으로 추가합니다.
    """
    if where == GRB.Callback.MIPSOL:
        # 현재 해에서 선택된 엣지(x[i, j] > 0.5)를 가져옵니다.
        vals = model.cbGetSolution(model._vars)
        selected_edges = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        
        n = model._n
        start_node = model._start_node

        # 시작 노드로부터 연결된 투어를 탐색
        tour = [start_node]
        current = start_node
        
        while True:
            # 현재 노드에서 나가는 엣지를 찾습니다.
            try:
                next_node = selected_edges.select(current, '*')[0][1]
                if next_node == start_node and len(tour) < n:
                    # 시작점으로 돌아왔지만 모든 노드를 방문하지 않음 (서브투어 발생)
                    break 
                
                if next_node not in tour:
                    tour.append(next_node)
                    current = next_node
                else:
                    # 이미 방문한 노드 (시작점 제외)로 돌아옴 (서브투어 발생)
                    break

                if len(tour) == n:
                    break # 모든 노드를 방문한 완전한 투어/경로
            
            except IndexError:
                # 다음 노드를 찾지 못함 (경로가 끊어짐)
                break
        
        # 만약 찾은 경로가 모든 노드를 포함하지 않는다면 (서브투어 발생)
        if len(tour) < n:
            # Gurobi에 서브투어를 제거하는 제약조건을 'lazy'하게 추가
            # S = tour (서브투어 노드 집합)
            # "S에서 S 외부로 나가는 엣지는 최소 1개 이상이어야 한다"
            model.cbLazy(
                gp.quicksum(model._vars[i, j]
                            for i in tour
                            for j in range(n) if j not in tour)
                >= 1
            )

def solve_gurobi(full_data, shopping_list, start_node_label, opts):
    """
    Gurobi로 '최적해'를 추론합니다.
    AI 모델과 동일하게, 미리 생성된 'subgraph' (instance)를 입력받습니다.

    Args:
        instance (torch_geometric.data.Data): _create_subgraph_from_labels로 생성된 서브그래프
        shopping_list (list): 상품 라벨 리스트 (인덱스를 라벨로 변환하기 위해 필요)

    Returns:
        tuple: (final_path_labels, optimal_cost)
               (최적 경로 라벨 리스트, 최적 경로의 총 비용)
               실패 시 (None, -1)
    """
    instance = _create_subgraph_from_labels(
        full_data, 
        shopping_list, 
        start_node_label,
        opts.device
    )

    # 1. Gurobi가 사용할 데이터 추출 (이제 인자에서 바로 가져옴)
    dist_matrix = instance.dist_matrix.cpu().numpy()
    start_idx = instance.start_idx.item()
    end_idx = instance.end_idx.item()
    n = instance.num_nodes
    
    nodes = list(range(n))
    
    # 2. Gurobi 모델 생성
    m = gp.Model("Shopping_TSP")
    m.setParam('OutputFlag', 0) # 로그 출력 끄기

    # 3. 변수(Variables) 생성 (x[i, j])
    x = m.addVars(n, n, vtype=GRB.BINARY, name='x')
    
    # 4. 목적 함수(Objective Function) 설정 (총 거리 최소화)
    m.setObjective(
        gp.quicksum(dist_matrix[i, j] * x[i, j] 
                    for i in nodes for j in nodes if i != j),
        GRB.MINIMIZE
    )

# 5. 제약 조건(Constraints) 설정
    m.addConstrs(x[i, i] == 0 for i in nodes) # i->i 이동 금지
    is_tour = (start_idx == end_idx) # 투어 문제인지 경로 문제인지 확인

    if is_tour:
        # [Case 1: 투어 문제 (start == end)]
        # (이 부분은 정상이므로 수정 없음)
        m.addConstrs(gp.quicksum(x[i, j] for i in nodes if i != j) == 1 for j in nodes) # In-degree
        m.addConstrs(gp.quicksum(x[j, i] for j in nodes if j != i) == 1 for i in nodes) # Out-degree
    
    else:
        # [Case 2: 경로 문제 (start != end)]
        # (시작점, 종료점 제약 조건은 정상이므로 수정 없음)
        m.addConstr(gp.quicksum(x[start_idx, j] for j in nodes if j != start_idx) == 1)
        m.addConstr(gp.quicksum(x[i, start_idx] for i in nodes if i != start_idx) == 0)
        m.addConstr(gp.quicksum(x[i, end_idx] for i in nodes if i != end_idx) == 1)
        m.addConstr(gp.quicksum(x[end_idx, j] for j in nodes if j != end_idx) == 0)
        
        # (e) 💥 중간 노드 제약 조건 수정
        for k in nodes:
            if k != start_idx and k != end_idx:
                
                # (e-1) Flow conservation: 들어온 만큼 나간다 (기존과 동일)
                m.addConstr(
                    gp.quicksum(x[i, k] for i in nodes if i != k) == 
                    gp.quicksum(x[k, j] for j in nodes if j != k),
                    name=f"flow_conv_{k}"
                )
                
                # (e-2) 💥 [핵심 수정] 
                #     중간 노드는 최대 1번만 방문(진입)할 수 있다.
                #     (e-1과 결합하면, 진입=1이면 진출=1, 진입=0이면 진출=0)
                m.addConstr(
                    gp.quicksum(x[i, k] for i in nodes if i != k) <= 1,
                    name=f"visit_max_once_{k}"
                )

    # 6. 최적화 실행 (LazyConstraint 콜백 사용)
    # (이하 6, 7, 8번은 이전에 수정한 'while' 루프 포함하여 그대로 사용)
    m._vars = x
    m._n = n
    m._start_node = start_idx
    m.Params.LazyConstraints = 1
    
    print(f"  [INFO] Gurobi 최적화 시작 (N={n})...")
    m.optimize(subtourelim) 
    
    # 7. 결과 추출 및 경로 변환
    if m.status == GRB.OPTIMAL:
        optimal_cost = m.ObjVal
        print(f"  [INFO] Gurobi 최적해 발견! (최적 비용: {optimal_cost:.4f})")
        
        # 경로 복원
        solution_edges = {}
        for i, j in x.keys():
            if x[i, j].X > 0.5:
                solution_edges[i] = j
        
        final_path_indices = []
        curr = start_idx
        while curr not in final_path_indices:
            final_path_indices.append(curr)

            if curr not in solution_edges:
                if len(final_path_indices) == n:
                    break # 모든 노드를 다 방문했으므로 정상 종료
                else: 
                    # E1이 아닌데 경로가 끊김 (오류)
                    print(f"  [오류] Gurobi: 경로 추적 실패. (현재 노드: {curr})")
                    return None, -1 
            
            curr = solution_edges[curr]
            
        # 8. 인덱스를 다시 상품 라벨로 변환
        final_path_labels = [shopping_list[i] for i in final_path_indices]
        return final_path_labels, optimal_cost

    else:
        print("  [오류] Gurobi: 최적해를 찾지 못했습니다.")
        return None, -1