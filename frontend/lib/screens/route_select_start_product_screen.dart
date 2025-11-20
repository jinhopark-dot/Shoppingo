import 'package:flutter/material.dart';
import '../models/product.dart';

/// 경로 재생성 1단계:
/// 전체 상품 중에서 "현재 위치를 대표할 물품" 하나 선택
class SelectStartProductScreen extends StatefulWidget {
  const SelectStartProductScreen({
    super.key,
    required this.idToProduct,
    required this.cart,
  });

  final Map<int, Product> idToProduct;
  final Map<int, int> cart; // 장바구니 (하이라이트 용도)

  @override
  State<SelectStartProductScreen> createState() =>
      _SelectStartProductScreenState();
}

class _SelectStartProductScreenState extends State<SelectStartProductScreen> {
  String _query = '';
  int? _selectedId;

  @override
  Widget build(BuildContext context) {
    final allProducts = widget.idToProduct.values.toList()
      ..sort((a, b) => a.name.compareTo(b.name));

    final filtered = _query.trim().isEmpty
        ? allProducts
        : allProducts
            .where((p) =>
                p.name.toLowerCase().contains(_query.toLowerCase()))
            .toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('현재 위치 선택'),
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
                '주변에 있는 물품 하나를 선택해\n현재 위치를 지정해 주세요.',
                textAlign: TextAlign.center, 
                style: TextStyle(
                  color: const Color.fromARGB(255, 0, 0, 0), 
                  fontSize: 18,
                  fontWeight: FontWeight.w600, 
                ),
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
                      final inCart = widget.cart.containsKey(p.id) &&
                          (widget.cart[p.id] ?? 0) > 0;
                      final selected = _selectedId == p.id;
                      return _buildRow(p, inCart, selected);
                    },
                  ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
          child: ElevatedButton(
            onPressed: _selectedId == null ? null : _onConfirm,
            child: const Text('현재 위치로 선택'),
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

  Widget _buildRow(Product p, bool inCart, bool selected) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          setState(() {
            _selectedId = p.id;
          });
        },
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: selected ? Colors.blue : const Color(0xFFE6E8EF),
              width: selected ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Radio<int>(
                value: p.id,
                groupValue: _selectedId,
                onChanged: (v) {
                  setState(() {
                    _selectedId = v;
                  });
                },
              ),
              const SizedBox(width: 4),
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
              if (inCart)
                const Padding(
                  padding: EdgeInsets.only(left: 4),
                  child: Text(
                    '장바구니',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.blue,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
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


  void _onConfirm() {
    if (_selectedId == null) return;
    Navigator.of(context).pop<int>(_selectedId);
  }
}
