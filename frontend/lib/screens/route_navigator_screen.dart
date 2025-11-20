// lib/screens/route_navigator_screen.dart
import 'package:flutter/material.dart';
import '../models/route_result.dart';
import '../models/product.dart';
import '../data/app_config.dart';
import '../models/store.dart';
import 'route_regenerate_screen.dart';
import 'route_select_start_product_screen.dart';
import 'package:photo_view/photo_view.dart';

class RouteNavigatorScreen extends StatefulWidget {
  const RouteNavigatorScreen({
    super.key,
    required this.store,
    required this.result,
    required this.idToProduct,
    required this.cart,
    required this.purchasedProductIds,
  });

  final Store store;
  final RouteResult result;
  final Map<int, Product> idToProduct;
  final Map<int, int> cart;

  /// 지금까지 "담은 것으로 확정된" 상품 id 집합 (B안)
  final Set<int> purchasedProductIds;

  @override
  State<RouteNavigatorScreen> createState() => _RouteNavigatorScreenState();
}

class _RouteNavigatorScreenState extends State<RouteNavigatorScreen> {
  late final List<String> nodes;
  late final List<List<int>> nodeProductIds;

  // 진행 중인 노드 인덱스 (물품 위치)
  int _currentIndex = 0;

  // 화면에 보여주는 노드 인덱스 (< > 로 이동)
  int _viewIndex = 0;

  // 노드별 체크된 상품 ID 집합
  late final List<Set<int>> _doneByNode;

  // 세션 전체에 공유되는 "이미 산 상품" id 집합 (B안)
  late Set<int> _purchasedProductIds;

  @override
  void initState() {
    super.initState();
    final r = widget.result;
    nodes = r.orderedNode;
    nodeProductIds = r.orderedProductIds;

    _currentIndex = 0;
    _viewIndex = 0;
    _doneByNode = List.generate(nodes.length, (_) => <int>{});

    // B안: 이전까지의 구매 내역을 복사해서 사용
    _purchasedProductIds = {...widget.purchasedProductIds};
  }

  String _absUrl(String pathOrUrl) {
    final s = pathOrUrl.trim();
    if (s.startsWith('http://') || s.startsWith('https://')) return s;
    return '${AppConfig.I.baseUrl}$s';
  }

  void _toggle(int productId) {
    setState(() {
      // 1) 현재 보고 있는 노드(_viewIndex)의 체크 상태 토글
      final set = _doneByNode[_viewIndex];
      final wasChecked = set.contains(productId);

      if (wasChecked) {
        set.remove(productId);
        // "담은 것으로 확정" 취소
        _purchasedProductIds.remove(productId);
      } else {
        set.add(productId);
        // 세션 전역에서 "이미 산 상품"으로 기록
        _purchasedProductIds.add(productId);
      }

      // 2) 자동 이동은 "진행 중인 노드" 화면일 때만 동작
      if (_viewIndex == _currentIndex &&
          _currentIndex < nodeProductIds.length) {
        final ids = nodeProductIds[_currentIndex];
        final currentSet = _doneByNode[_currentIndex];

        final allDone =
            ids.isEmpty || ids.every((id) => currentSet.contains(id));

        if (allDone && _currentIndex < nodes.length - 1) {
          _currentIndex += 1;
          _viewIndex = _currentIndex; // 화면도 같이 다음 노드로 이동
        }
      }
    });
  }

  void _skipCurrentNode() {
    if (_currentIndex >= nodes.length - 1) return;

    setState(() {
      _currentIndex += 1;
      _viewIndex = _currentIndex;
    });
  }

  /// 경로 재생성 버튼 핸들러 (B안 전체 흐름)
  /// 1) 아직 안 산 물품이 하나도 없으면 토스트만 띄우고 종료
  /// 2) 전체 상품 중에서 "현재 위치 물품"을 고르는 화면으로 이동
  /// 3) 장바구니 수정 + 새 경로 생성 화면으로 이동
  /// 4) 새 RouteNavigatorScreen 으로 갈아끼우기
  Future<void> _onRegeneratePressed() async {
    // 1) 남은 물품이 있는지 확인
    final hasRemaining = widget.cart.entries.any(
      (e) => e.value > 0 && !_purchasedProductIds.contains(e.key),
    );

    if (!hasRemaining) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('안내할 남은 물품이 없습니다.')),
      );
      return;
    }

    // 2) 전체 상품 중에서 "현재 위치"가 될 물품 선택
    final startProductId = await Navigator.of(context).push<int?>(
      MaterialPageRoute(
        builder: (_) => SelectStartProductScreen(
          idToProduct: widget.idToProduct,
          cart: widget.cart,
        ),
      ),
    );

    if (startProductId == null) {
      // 사용자가 뒤로가기 등으로 취소
      return;
    }

    // 3) 장바구니 수정 + 경로 재생성 화면
    final regenResult =
        await Navigator.of(context).push<RouteRegenerateResult?>(
      MaterialPageRoute(
        builder: (_) => RouteRegenerateScreen(
          store: widget.store,
          idToProduct: widget.idToProduct,
          initialCart: widget.cart,
          purchasedProductIds: _purchasedProductIds,
          startProductId: startProductId,
        ),
      ),
    );

    // 사용자가 취소
    if (regenResult == null) return;
    if (!mounted) return;

    // 4) 새 경로 안내 화면으로 교체
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => RouteNavigatorScreen(
          store: widget.store,
          result: regenResult.routeResult,
          idToProduct: widget.idToProduct,
          cart: regenResult.updatedCart,
          purchasedProductIds: _purchasedProductIds, // B안 유지
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final totalNodes = nodes.length;

    // 물품 위치(진행 중인 노드)와, 화면에 보여주는 노드 표시용 숫자
    final currentDisplay =
        (_currentIndex < totalNodes) ? '${_currentIndex + 1}' : '완료';
    final viewDisplay =
        (_viewIndex < totalNodes) ? '${_viewIndex + 1}' : '완료';

    final ids = _viewIndex < nodeProductIds.length
        ? nodeProductIds[_viewIndex]
        : const <int>[];

    return Scaffold(
      appBar: AppBar(
        title: const Text('경로 안내'),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // 상단: 경로 이미지 (확대,축소,회전 기능 추가)
          AspectRatio(
            aspectRatio: 12/13,
            child: ClipRect( // 회전 시 이미지가 영역 밖으로 튀어나가지 않게 자름
              child: PhotoView(
                imageProvider: NetworkImage(_absUrl(widget.result.routeImageUrl)),
                enableRotation: true, //회전 기능

                backgroundDecoration: const BoxDecoration(
                  color: Colors.transparent, 
                ),
                minScale: PhotoViewComputedScale.contained, 
                maxScale: PhotoViewComputedScale.covered * 3.0,

                initialScale: PhotoViewComputedScale.contained,

                errorBuilder: (context, error, stackTrace) {
                  return Container(
                    color: Colors.grey.shade200,
                    alignment: Alignment.center,
                    child: const Icon(Icons.broken_image, size: 64),
                  );
                },
                
                loadingBuilder: (context, event) => const Center(
                  child: CircularProgressIndicator(),
                ),
              ),
            ),
          ),
          const Divider(height: 1),

          // 현재 노드 타이틀
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Row(
              children: [
                // 물품 위치: X  → 탭하면 현재 진행 노드로 화면 복귀
                GestureDetector(
                  onTap: () {
                    setState(() {
                      _viewIndex = _currentIndex;
                    });
                  },
                  child: Row(
                    children: [
                      Text(
                        '물품 위치: ',
                        style: TextStyle(
                          color: Colors.grey.shade700,
                          fontSize: 14,
                        ),
                      ),
                      Text(
                        currentDisplay,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),

                // < 노드 3/6 >
                IconButton(
                  icon: const Icon(Icons.chevron_left),
                  onPressed: _viewIndex > 0
                      ? () {
                          setState(() {
                            _viewIndex -= 1;
                          });
                        }
                      : null,
                ),
                Text(
                  '노드 $viewDisplay/$totalNodes',
                  style: TextStyle(color: Colors.grey.shade700, fontSize: 14),
                ),
                IconButton(
                  icon: const Icon(Icons.chevron_right),
                  onPressed: _viewIndex < totalNodes - 1
                      ? () {
                          setState(() {
                            _viewIndex += 1;
                          });
                        }
                      : null,
                ),
              ],
            ),
          ),

          // 노드별 품목 리스트
          Expanded(
            child: ids.isEmpty
                ? Center(
                    child: Text(
                      '이 노드에 담을 품목이 없습니다.',
                      style: TextStyle(color: Colors.grey.shade600),
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    itemCount: ids.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (_, i) {
                      final pid = ids[i];
                      final p = widget.idToProduct[pid];
                      final name = p?.name ?? '상품 #$pid';
                      final checked = _doneByNode[_viewIndex].contains(pid);
                      final qty = widget.cart[pid] ?? 1; // 장바구니 기준 수량

                      return _ChecklistTile(
                        title: name,
                        subtitle: () {
                          final price = p?.price;
                          if (price == null) return null;
                          return '${price.round()}원';
                        }(),
                        imageUrl: p?.imageUrl,
                        quantity: qty,
                        checked: checked,
                        onTap: () => _toggle(pid),
                      );
                    },
                  ),
          ),

          // 하단 버튼들
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
              child: Row(
                children: [
                  // 왼쪽: 경유 노드 수
                  Text(
                    '경유 노드 수: ${nodes.length}',
                    style: TextStyle(color: Colors.grey.shade700),
                  ),
                  const Spacer(),

                  // 가운데: 경로 재생성 버튼 (회전 화살표)
                  ElevatedButton.icon(
                    onPressed: _onRegeneratePressed,
                    icon: const Icon(Icons.autorenew),
                    label: const Text('경로 재생성'),
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(0, 40),
                      padding:
                          const EdgeInsets.symmetric(horizontal: 12),
                    ),
                  ),

                  const SizedBox(width: 8),

                  // 오른쪽: 넘기기 버튼 (>|)
                  ElevatedButton.icon(
                    onPressed: _currentIndex < nodes.length - 1
                        ? _skipCurrentNode
                        : null,
                    icon: const Icon(Icons.skip_next),
                    label: const Text('넘기기'),
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(0, 40),
                      padding:
                          const EdgeInsets.symmetric(horizontal: 12),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ChecklistTile extends StatelessWidget {
  const _ChecklistTile({
    required this.title,
    this.subtitle,
    this.imageUrl,
    required this.quantity,
    required this.checked,
    required this.onTap,
  });

  final String title;
  final String? subtitle;
  final String? imageUrl;
  final int quantity;
  final bool checked;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final borderColor = checked ? Colors.green : Colors.transparent;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      elevation: 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          decoration: BoxDecoration(
            border: Border.all(color: borderColor, width: 2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            children: [
              // 상품 썸네일
              if (imageUrl != null && imageUrl!.isNotEmpty) ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.network(
                    imageUrl!,
                    width: 48,
                    height: 48,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const Icon(
                      Icons.image_not_supported,
                      color: Colors.grey,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
              ],

              // 이름 / 가격
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    if (subtitle != null)
                      Text(
                        subtitle!,
                        style: TextStyle(
                          color: Colors.grey.shade600,
                          fontSize: 12,
                        ),
                      ),
                  ],
                ),
              ),

              const SizedBox(width: 8),

              // 수량 표시: x3 (1개면 생략)
              if (quantity > 1)
                Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: Text(
                    'x$quantity',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF375BE8),
                    ),
                  ),
                ),

              // 체크 아이콘
              Icon(
                checked ? Icons.check_circle : Icons.check_circle_outline,
                color: checked ? Colors.green : Colors.grey,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
