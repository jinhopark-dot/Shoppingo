import 'package:flutter/material.dart';
import '../models/store.dart';
import '../models/product.dart';
import '../models/route_result.dart';
import '../services/store_service.dart';

/// 재생성 결과: 새 RouteResult + 수정된 장바구니
class RouteRegenerateResult {
  RouteRegenerateResult({
    required this.routeResult,
    required this.updatedCart,
  });

  final RouteResult routeResult;
  final Map<int, int> updatedCart;
}

/// 경로 재생성 2단계:
/// - 장바구니 수정
/// - "아직 안 산 상품"만 서버에 보내 새 경로 생성
class RouteRegenerateScreen extends StatefulWidget {
  const RouteRegenerateScreen({
    super.key,
    required this.store,
    required this.idToProduct,
    required this.initialCart,
    required this.purchasedProductIds,
    required this.startProductId,
  });

  final Store store;
  final Map<int, Product> idToProduct;
  final Map<int, int> initialCart;
  final Set<int> purchasedProductIds;
  final int startProductId;

  @override
  State<RouteRegenerateScreen> createState() => _RouteRegenerateScreenState();
}

class _RouteRegenerateScreenState extends State<RouteRegenerateScreen> {
  late Map<int, int> _cart; // 수정 가능한 장바구니
  String _query = '';
  bool _loading = false;

  final _service = StoreService();

  @override
  void initState() {
    super.initState();
    _cart = Map<int, int>.from(widget.initialCart);
  }

  @override
  Widget build(BuildContext context) {
    final products = widget.idToProduct.values.toList()
      ..sort((a, b) => a.name.compareTo(b.name));

    final filtered = _query.trim().isEmpty
        ? products
        : products
            .where((p) =>
                p.name.toLowerCase().contains(_query.toLowerCase()))
            .toList();

    final startProduct = widget.idToProduct[widget.startProductId];

    return Scaffold(
      appBar: AppBar(
        title: const Text('경로 재생성'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
              decoration: BoxDecoration(
                color: Colors.white, // 1. 흰색 배경
                borderRadius: BorderRadius.circular(12), 
                border: Border.all(color: Colors.grey.shade300), 
              ),
              child: Text(
                '장바구니를 수정한 뒤\n남은 물품에 대한 경로를 다시 생성합니다.',
                textAlign: TextAlign.center, 
                style: TextStyle(
                  color: const Color.fromARGB(255, 0, 0, 0), 
                  fontSize: 18,
                  fontWeight: FontWeight.w600, 
                ),
              ),
            ),
          ),
          if (startProduct != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '현재 위치: ${startProduct.name}',
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
          _buildSearchField(),
          const SizedBox(height: 8),
          Expanded(
            child: filtered.isEmpty
                ? const Center(child: Text('표시할 상품이 없습니다.'))
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
                    itemCount: filtered.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (context, index) {
                      final p = filtered[index];
                      final qty = _cart[p.id] ?? 0;
                      return _buildProductRow(p, qty);
                    },
                  ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  '장바구니 상품 수: ${_cart.length}',
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey.shade700,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: _loading ? null : _onConfirmPressed,
                child: _loading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('경로 재생성'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSearchField() {
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

  Widget _buildProductRow(Product p, int qty) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          setState(() {
            final current = _cart[p.id] ?? 0;
            final newQty = current + 1;
            _cart[p.id] = newQty;
          });
        },
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: const Color(0xFFE6E8EF),
              width: 1,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              _thumb(p.imageUrl),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      p.name,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      p.price != null
                          ? '${_formatPrice(p.price!)}원'
                          : '가격 정보 없음',
                      style: const TextStyle(
                        fontSize: 13,
                        color: Colors.black54,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              if (qty > 0)
                Text(
                  'x$qty',
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: Colors.blue,
                  ),
                ),
              IconButton(
                icon: const Icon(Icons.edit),
                onPressed: () async {
                  final newQty =
                      await _showQuantityDialog(product: p, initial: qty);
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

  Widget _thumb(String? url) {
    if (url == null || url.isEmpty) {
      return Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          color: const Color(0xFFEDEFF5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: const Icon(Icons.image_not_supported, color: Colors.black45),
      );
    }
    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.network(
        url,
        width: 48,
        height: 48,
        fit: BoxFit.cover,
        errorBuilder: (_, __, ___) => Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: const Color(0xFFEDEFF5),
            borderRadius: BorderRadius.circular(8),
          ),
          child:
              const Icon(Icons.image_not_supported, color: Colors.black45),
        ),
      ),
    );
  }

  String _formatPrice(int v) {
    final s = v.toString();
    final buf = StringBuffer();

    // 왼쪽에서 오른쪽으로 가면서 "뒤에서 몇 번째 자리인지" 기준으로 콤마 삽입
    for (int i = 0; i < s.length; i++) {
      buf.write(s[i]);

      // 남은 자리 수
      final remaining = s.length - i - 1;
      if (remaining > 0 && remaining % 3 == 0) {
        buf.write(',');
      }
    }
    return buf.toString();
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

  Future<void> _onConfirmPressed() async {
    // "아직 안 산 상품"만 서버에 보낼 대상
    final productIds = <int>[];
    _cart.forEach((productId, qty) {
      if (qty > 0 && !widget.purchasedProductIds.contains(productId)) {
        productIds.add(productId);
      }
    });

    if (productIds.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('안내할 남은 물품이 없습니다.')),
      );
      return;
    }

    setState(() => _loading = true);

    try {
      final result = await _service.createRouteResult(
        storeId: widget.store.id,
        productIds: productIds,
        // backend: start_node에 "상품 id" 문자열을 넘기면 노드로 변환
        startNode: widget.startProductId.toString(),
      );

      if (!mounted) return;
      Navigator.of(context).pop(
        RouteRegenerateResult(
          routeResult: result,
          updatedCart: Map<int, int>.from(_cart),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('경로 재생성 실패: $e')),
      );
    }
  }
}
