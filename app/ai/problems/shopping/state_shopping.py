import torch
from typing import NamedTuple
from ...utils.boolmask import mask_long2bool, mask_long_scatter


class StateShopping(NamedTuple):
    """
    쇼핑 경로 최적화 문제의 상태
    """
    
    # Fixed input
    dist_matrix: torch.Tensor  # (batch, n_nodes, n_nodes)
    start_idx: torch.Tensor    # (batch,)
    end_idx: torch.Tensor      # (batch,)
    
    # Beam search용 ID
    ids: torch.Tensor          # (batch, 1)
    
    # State
    prev_a: torch.Tensor       # (batch, 1) 이전 노드
    visited_: torch.Tensor     # (batch, 1, n_nodes) 방문 마스크
    lengths: torch.Tensor      # (batch, 1) 누적 거리
    i: torch.Tensor            # (1,) 스텝 카운터
    
    # ✅ 1. 부분 경로 전체를 저장할 필드 추가
    tour_: torch.Tensor        # (batch, i) 현재까지의 경로
    
    @property
    def visited(self):
        """방문 마스크 반환"""
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.dist_matrix.size(-2))
    
    def __getitem__(self, key):
        """Beam search를 위한 인덱싱"""
        assert torch.is_tensor(key) or isinstance(key, slice)
        return self._replace(
            ids=self.ids[key],
            start_idx=self.start_idx[key],
            end_idx=self.end_idx[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            tour_=self.tour_[key] # ✅ getitem에도 추가
        )
    
    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        """
        초기 상태 생성
        
        시작 노드를 이미 방문한 것으로 표시
        """
        dist_matrix = input['dist_matrix']
        batch_size, n_nodes, _ = dist_matrix.size()
        
        start_idx = input['start_idx']  # (batch,)
        end_idx = input['end_idx']      # (batch,)
        
        # 시작 노드
        prev_a = start_idx.unsqueeze(-1)  # (batch, 1)
        tour_ = prev_a.clone()
        
        # 방문 마스크 초기화
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(batch_size, 1, n_nodes, dtype=torch.uint8, device=dist_matrix.device)
        else:
            visited_ = torch.zeros(batch_size, 1, (n_nodes + 63) // 64, dtype=torch.int64, device=dist_matrix.device)
        
        # 🆕 시작 노드를 방문한 것으로 표시
        if visited_dtype == torch.uint8:
            visited_ = visited_.scatter(-1, prev_a.unsqueeze(-1), 1)
        else:
            visited_ = mask_long_scatter(visited_, prev_a)
        
        return StateShopping(
            dist_matrix=dist_matrix,
            start_idx=start_idx,
            end_idx=end_idx,
            ids=torch.arange(batch_size, dtype=torch.int64, device=dist_matrix.device).unsqueeze(-1),
            prev_a=prev_a,
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=dist_matrix.device),
            i=torch.zeros(1, dtype=torch.int64, device=dist_matrix.device),
            tour_=tour_
        )
    
    def get_current_node(self):
        """현재 노드 반환"""
        return self.prev_a
    
    def get_end_node(self):
        """목적지 노드 반환"""
        return self.end_idx.unsqueeze(-1)  # (batch, 1)
    
    def all_finished(self):
        """
        모든 노드를 방문했는지 확인
        """
        return self.i.item() >= self.dist_matrix.size(1) - 1
    
    def get_mask(self):
        """
        🔑 핵심: 방문 불가능한 노드 마스킹
        
        규칙:
        1. 이미 방문한 노드는 방문 불가
        2. 끝 노드는 모든 다른 노드를 방문한 후에만 선택 가능
        
        Returns:
            mask: (batch, 1, n_nodes) bool tensor
                  True = 방문 불가, False = 방문 가능
        """
        batch_size, _, n_nodes = self.visited.size()
        
        # 1. 기본 마스크: 이미 방문한 노드
        mask = self.visited > 0  # (batch, 1, n_nodes)
        
        # 2. 끝 노드 특별 처리
        end_mask = self._get_end_node_mask()  # (batch, 1, n_nodes)
        
        # 최종 마스크 = 방문한 노드 OR 끝 노드 (조건부)
        mask = mask | end_mask
        
        return mask
    
    def _get_end_node_mask(self):
        """
        끝 노드 마스킹 로직
        
        끝 노드는 마지막에만 선택 가능:
        - 방문해야 할 노드 수 = n_nodes - 1 (시작 노드 제외)
        - 현재 스텝 < n_nodes - 1 이면 끝 노드 마스킹
        - 현재 스텝 == n_nodes - 1 이면 끝 노드만 선택 가능
        
        Returns:
            end_mask: (batch, 1, n_nodes) bool tensor
        """
        batch_size, _, n_nodes = self.visited.size()
        device = self.visited.device
        
        # 끝 노드 위치 마스크 생성
        end_node_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool, device=device)
        
        # 각 배치별로 끝 노드 위치 마킹
        batch_idx = torch.arange(batch_size, device=device)
        end_node_mask[batch_idx, 0, self.end_idx] = True
        
        # 🔑 핵심 로직
        # 아직 모든 노드를 방문하지 않았으면 끝 노드 마스킹
        # (n_nodes - 1: 시작 노드 제외)
        current_step = self.i.item()
        
        if current_step < n_nodes - 2:
            # 아직 끝 노드 선택 불가
            return end_node_mask
        else:
            # 마지막 스텝: 끝 노드만 선택 가능
            # 따라서 끝 노드는 마스킹하지 않음
            return torch.zeros_like(end_node_mask)
    
    def update(self, selected):
        """
        선택한 노드로 상태 업데이트
        """
        prev_a = selected.unsqueeze(-1)  # (batch, 1)
        
        # 이동 거리 계산
        batch_size = self.dist_matrix.size(0)
        batch_idx = torch.arange(batch_size, device=self.dist_matrix.device)
        distances = self.dist_matrix[batch_idx, self.prev_a.squeeze(-1), selected]
        lengths = self.lengths + distances.unsqueeze(-1)
        
        # 방문 마킹
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a.unsqueeze(-1), 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)
        
        # ✅ 3. tour_ 업데이트 (기존 경로에 선택된 노드 추가)
        tour_ = torch.cat((self.tour_, selected.unsqueeze(-1)), dim=1)

        return self._replace(
            prev_a=prev_a,
            visited_=visited_,
            lengths=lengths,
            i=self.i + 1,
            tour_=tour_ # tour_ 업데이트
        )
    
    def construct_solutions(self, actions):
        """
        행동 시퀀스를 솔루션으로 변환
        """
        return actions
    
    # state_shopping.py에 추가

    def get_finished(self):
        """
        각 인스턴스가 완료되었는지 반환
        
        Returns:
            finished: (batch,) 0 = 진행 중, 1 = 완료
        """
        n_nodes = self.dist_matrix.size(1)
        finished = (self.i >= n_nodes - 1).long()
        return finished.expand(self.ids.size(0))