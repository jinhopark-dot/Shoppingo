// lib/screens/shopping_list_screen.dart
import 'package:flutter/material.dart';
import '../models/store.dart';
import '../models/product.dart';
import '../services/store_service.dart';
import '../constants.dart';
import 'route_navigator_screen.dart';

class ShoppingListScreen extends StatefulWidget {
  const ShoppingListScreen({super.key, required this.store,});
  final Store store; // 어떤 매장의 쇼핑리스트인지 명확히 보이도록 전체 Store 전달

  @override
  State<ShoppingListScreen> createState() => _ShoppingListScreenState();
}

class _ShoppingListScreenState extends State<ShoppingListScreen>
    with SingleTickerProviderStateMixin {
  // 장바구니(Map<int,int> _cart = {productId: qty}) → product_ids 리스트(수량만큼 중복 포함)
  List<int> _buildProductIdListFromCart() {
    final List<int> out = [];
    for (final e in _cart.entries) {
      out.addAll(List.filled(e.value, e.key));
    }
    return out;
  }

  // id → Product 맵 (경로 화면 하단 품목명/가격 표시에 사용)
  Map<int, Product> _buildIdToProductMap() {
    // 당신 파일에선 재고 전체가 _all 이므로 그걸 이용
    return { for (final p in _all) p.id: p };
  }

  final _service = StoreService();           // 이미 다른 곳에 있으면 중복 생성하지 마세요.
  String _startNode = kDefaultStartNode;     // 최초 S1, 재생성 시 갱신

  // 카탈로그/인벤토리 리스트를 이미 갖고 있는 컬렉션명으로 바꾸세요.

  late final TabController _tab;

  bool _loading = true;
  String? _error;
  List<Product> _all = const [];
  String _query = '';

  // 장바구니: productId -> qty
  final Map<int, int> _cart = {};

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final items = await _service.fetchInventory(widget.store.id);
      setState(() {
        _all = items;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _tab.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final filtered = _query.trim().isEmpty
        ? _all
        : _all.where((p) => p.name.toLowerCase().contains(_query.toLowerCase())).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('쇼핑 리스트 작성'),
        bottom: TabBar(
          controller: _tab,
          tabs: const [
            Tab(text: '매장 쇼핑'),
            Tab(text: '장바구니'),
          ],
        ),
      ),
      body: Column(
        children: [
          _storeHeader(widget.store),
          _searchField(),
          const SizedBox(height: 8),
          Expanded(
            child: TabBarView(
              controller: _tab,
              children: [
                _buildStoreTab(filtered),
                _buildCartTab(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // 상단 매장 "버튼처럼 보이는" 정보 카드(동작 없음)
  Widget _storeHeader(Store s) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFFE6E8EF)),
        ),
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            _storeThumb(s.imageUrl),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(s.name,
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w700)),
                  if ((s.address ?? '').isNotEmpty)
                    Text(s.address!,
                        style: const TextStyle(
                            fontSize: 13, color: Colors.black54)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 검색창(물품 검색)
  Widget _searchField() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: TextField(
        decoration: InputDecoration(
          hintText: '물품 검색',
          prefixIcon: const Icon(Icons.search),
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
            borderSide: BorderSide.none,
          ),
          contentPadding:
              const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
        ),
        onChanged: (v) => setState(() => _query = v),
      ),
    );
  }

  // 매장 쇼핑 탭
  Widget _buildStoreTab(List<Product> items) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('재고 목록을 불러오지 못했습니다.'),
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 12),
              ElevatedButton(onPressed: _load, child: const Text('다시 시도')),
            ],
          ),
        ),
      );
    }
    if (items.isEmpty) {
      return const Center(child: Text('표시할 상품이 없습니다.'));
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      itemCount: items.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, i) {
        final p = items[i];
        final qty = _cart[p.id] ?? 0;
        return _productCard(p, qty);
      },
    );
  }

  // 상품 카드 UI
  // 요구사항 반영:
  // 1) 카드 전체 탭 시 qty+1
  // 2) 카트 버튼 탭 시 '원하는 수량 입력' 팝업
  // 3) 카트 아이콘 왼쪽에 담은 갯수 숫자 크게 표시(좌정렬)
  Widget _productCard(Product p, int qty) {
    final priceText =
        (p.price != null) ? _formatWon(p.price!) : '가격 정보 없음';

    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          // 1) 카드 탭마다 1개 담기
          setState(() {
            final cur = _cart[p.id] ?? 0;
            _cart[p.id] = cur + 1;
          });
        },
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: const Color(0xFFE6E8EF)),
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              _productThumb(p.imageUrl),
              const SizedBox(width: 12),
              // 이름/가격
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(p.name,
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 6),
                    Text(priceText,
                        style: const TextStyle(
                            fontSize: 14, color: Colors.black54)),
                  ],
                ),
              ),
              const SizedBox(width: 8),

              // 3) 담은 갯수 숫자(왼쪽 정렬, 크고 굵게)
              // 카트 버튼 높이와 시각적으로 맞추기 위해 SizedBox 고정폭 사용
              SizedBox(
                width: 40,
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    qty > 0 ? '$qty' : '',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: kAppBlue,
                    ),
                  ),
                ),
              ),

              // 2) 카트 버튼(팝업으로 수량 직접 입력/수정)
              IconButton(
                tooltip: '수량 지정',
                icon: const Icon(Icons.shopping_cart),
                onPressed: () async {
                  final newQty = await _showQuantityDialog(
                    product: p,
                    initial: qty,
                  );
                  if (newQty == null) return;
                  setState(() {
                    if (newQty <= 0) {
                      _cart.remove(p.id);
                    } else {
                      _cart[p.id] = newQty;
                    }
                  });
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<int?> _showQuantityDialog({
    required Product product,
    required int initial,
  }) async {
    final controller = TextEditingController(
      text: initial > 0 ? '$initial' : '1',
    );
    return showDialog<int>(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          title: Text('${product.name} 수량'),
          content: TextField(
            controller: controller,
            keyboardType: TextInputType.number,
            autofocus: true,
            decoration: const InputDecoration(
              hintText: '수량을 입력하세요 (0 입력 시 장바구니에서 제거)',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text('취소'),
            ),
            ElevatedButton(
              onPressed: () {
                final raw = controller.text.trim();
                final val = int.tryParse(raw);
                if (val == null || val < 0) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('0 이상의 정수를 입력하세요.')),
                  );
                  return;
                }
                Navigator.of(ctx).pop(val);
              },
              child: const Text('확인'),
            ),
          ],
        );
      },
    );
  }

  Widget _productThumb(String? url) {
    final fixed = _fixImageUrl(url);
    if (fixed != null && fixed.isNotEmpty) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.network(
          fixed,
          width: 64,
          height: 64,
          fit: BoxFit.cover,
          errorBuilder: (_, __, ___) => _fallbackThumb(),
        ),
      );
    }
    return _fallbackThumb();
  }

  // 매장 헤더용 썸네일 (48x48)
  Widget _storeThumb(String? url) {
    final fixed = _fixImageUrl(url);
    if (fixed != null && fixed.isNotEmpty) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.network(
          fixed,
          width: 48,
          height: 48,
          fit: BoxFit.cover,
          errorBuilder: (_, __, ___) => _fallbackStoreIcon(),
        ),
      );
    }
    return _fallbackStoreIcon();
  }

  Widget _fallbackStoreIcon() => Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          color: const Color(0xFFEDEFF5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: const Icon(Icons.store, color: Colors.black54),
      );
  
    // 에뮬레이터에서 127.0.0.1 / localhost → 10.0.2.2
  String? _fixImageUrl(String? url) {
    if (url == null || url.isEmpty) return null;
    var u = url;
    if (u.startsWith('http://127.0.0.1')) {
      u = u.replaceFirst('http://127.0.0.1', 'http://10.0.2.2');
    } else if (u.startsWith('http://localhost')) {
      u = u.replaceFirst('http://localhost', 'http://10.0.2.2');
    }
    return u;
  }



  Widget _fallbackThumb() => Container(
        width: 64,
        height: 64,
        decoration: BoxDecoration(
          color: const Color(0xFFEDEFF5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: const Icon(Icons.image_not_supported, color: Colors.black45),
      );

  // 장바구니 탭
  Widget _buildCartTab() {
    if (_cart.isEmpty) {
      return const Center(child: Text('장바구니가 비어 있습니다.'));
    }

    final productsById = {for (final p in _all) p.id: p};
    final entries = _cart.entries.toList();

    int total = 0;
    for (final e in entries) {
      final p = productsById[e.key];
      if (p?.price != null) {
        total += (p!.price! * e.value);
      }
    }

    return Column(
      children: [
        Expanded(
          child: ListView.separated(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            itemCount: entries.length,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, i) {
              final e = entries[i];
              final p = productsById[e.key];
              if (p == null) return const SizedBox.shrink();
              return Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: const Color(0xFFE6E8EF)),
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    _productThumb(p.imageUrl),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(p.name,
                              style: const TextStyle(
                                  fontSize: 16, fontWeight: FontWeight.w600)),
                          const SizedBox(height: 6),
                          Text(
                            (p.price != null)
                                ? _formatWon(p.price!)
                                : '가격 정보 없음',
                            style: const TextStyle(
                                fontSize: 14, color: Colors.black54),
                          ),
                        ],
                      ),
                    ),
                    // 현재 수량 표시
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: Text(
                        'x${e.value}',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: kAppBlue,
                        ),
                      ),
                    ),
                    // 수량 편집(팝업)
                    IconButton(
                      tooltip: '수량 변경',
                      icon: const Icon(Icons.edit),
                      onPressed: () async {
                        final newQty = await _showQuantityDialog(
                          product: p,
                          initial: e.value,
                        );
                        if (newQty == null) return;
                        setState(() {
                          if (newQty <= 0) {
                            _cart.remove(p.id);
                          } else {
                            _cart[p.id] = newQty;
                          }
                        });
                      },
                    ),
                    // 삭제 버튼 유지
                    IconButton(
                      tooltip: '제거',
                      icon: const Icon(Icons.delete_outline),
                      onPressed: () => setState(() => _cart.remove(p.id)),
                    ),
                  ],
                ),
              );
            },
          ),
        ),
        Container(
          color: Colors.white,
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  '합계: ${_formatWon(total)}',
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w700),
                ),
              ),
              // 4) 결제 → 최적경로생성
              ElevatedButton(
                onPressed: () async {
                // 3.5 요구사항: 장바구니 공집합이면 안내 후 매장 쇼핑 탭으로 복귀
                  if (_cart.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('쇼핑 리스트가 비어 있습니다. 담을 물품을 추가해 주세요.')),
                    );
                    _tab.animateTo(0); // 매장 쇼핑 탭으로 이동
                    return;
                  }
                  // 장바구니에 항목이 있을 때만 실제 최적 경로 생성 흐름 진행
                  // TODO: 최적 경로 API 호출/화면 이동 로직 연결
                  final productIds = _buildProductIdListFromCart();
                  final startNode = _startNode;

                  showDialog(
                    context: context,
                    barrierDismissible: false,
                    builder: (_) => const Center(child: CircularProgressIndicator()),
                  );

                  try {
                    final routeResult = await _service.createRouteResult(
                      storeId: widget.store.id,
                      productIds: productIds,
                      startNode: startNode,
                    );
                    
                    if (!mounted) return;
                    Navigator.of(context).pop(); // 로딩 닫기
                    
                    final idToProduct = _buildIdToProductMap();
                    
                    final nextStart = await Navigator.of(context).push<String?>(
                      MaterialPageRoute(
                        builder: (_) => RouteNavigatorScreen(
                          store: widget.store,
                          result: routeResult,
                          idToProduct: idToProduct,
                          cart: Map<int, int>.from(_cart),
                          purchasedProductIds: <int>{}, // 첫 경로 진입 시에는 아직 아무것도 안 샀다
                        ),
                      ),
                    );

                    if (nextStart != null && nextStart.isNotEmpty) {
                      setState(() => _startNode = nextStart);
                    }

                  } catch (e) {
                    if (!mounted) return;
                    Navigator.of(context).pop(); // 로딩 닫기
                    ScaffoldMessenger.of(context)
                        .showSnackBar(SnackBar(content: Text('경로 생성 실패: $e')));
                  }
                },
                child: const Text('최적경로생성'),
              )
            ],
          ),
        ),
      ],
    );
  }

  String _formatWon(int v) {
    final s = v.toString();
    final buf = StringBuffer();
    for (int i = 0; i < s.length; i++) {
      final idx = s.length - 1 - i;
      buf.write(s[idx]);
      if ((i + 1) % 3 == 0 && idx != 0) buf.write(',');
    }
    return '${buf.toString().split('').reversed.join()}원';
  }
}
