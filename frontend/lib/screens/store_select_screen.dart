// lib/screens/store_select_screen.dart
import 'package:flutter/material.dart';
import '../models/store.dart';
import '../services/store_service.dart';
import './settings_screen.dart';
import './shopping_list_screen.dart';

class StoreSelectScreen extends StatefulWidget {
  const StoreSelectScreen({super.key});

  @override
  State<StoreSelectScreen> createState() => _StoreSelectScreenState();
}

class _StoreSelectScreenState extends State<StoreSelectScreen> {
  final _service = StoreService();

  bool _loading = true;
  String? _error;
  List<Store> _stores = const [];
  String _searchQuery = '';
  int? _selectedId;

  @override
  void initState() {
    super.initState();
    _loadStores();
  }

  Future<void> _loadStores() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final list = await _service.fetchStores();
      setState(() {
        _stores = list;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
    if (!mounted) return;
  }

  @override
  Widget build(BuildContext context) {
    // Scaffold를 한 번만 만들고, 본문만 분기
    return Scaffold(
      appBar: _buildAppBar(context),
      body: _buildBody(),
      backgroundColor: const Color(0xFFF5F6F8),
    );
  }

  PreferredSizeWidget _buildAppBar(BuildContext context) => AppBar(
    title: const Text('매장 선택'),
    centerTitle: true,
    actions: [
      IconButton(
        icon: const Icon(Icons.settings),
        tooltip: '설정',
        onPressed: () async {
          final saved = await Navigator.of(context).push<bool>(
            MaterialPageRoute(builder: (_) => const SettingsScreen()),
          );
          if (!mounted) return;      // await 직후
          if (saved == true) {
            _loadStores();           // 필요 시 재조회
          }
        },
      ),
    ],
  );

  Widget _buildBody() {
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
              const Text('매장 목록을 불러오지 못했습니다.'),
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _loadStores,
                child: const Text('다시 시도'),
              ),
            ],
          ),
        ),
      );
    }

    final view = _searchQuery.trim().isEmpty
        ? _stores
        : _stores
            .where((s) =>
                s.name.toLowerCase().contains(_searchQuery.toLowerCase()))
            .toList(growable: false);

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _searchField(),
          const SizedBox(height: 12),
          Expanded(
            child: ListView.separated(
              itemCount: view.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, i) {
                final s = view[i];
                final selected = s.id == _selectedId;
                return ListTile(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  tileColor: Colors.white,
                  selected: selected,
                  selectedTileColor: const Color(0x11375BE8),
                  leading: _storeThumb(s.imageUrl),
                  title: Text(s.name,
                      style: const TextStyle(fontWeight: FontWeight.w600)),
                  subtitle: (s.address != null && s.address!.isNotEmpty)
                      ? Text(s.address!)
                      : null,
                  trailing:
                      selected ? const Icon(Icons.check) : null,
                  onTap: () => setState(() {
                    _selectedId = selected ? null : s.id;
                  }),
                );
              },
            ),
          ),
          Align(
            alignment: Alignment.centerRight,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24)),
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              ),
              onPressed: () async {
                if (_selectedId == null) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('매장을 선택해 주세요.')),
                  );
                  return;
                }

                final sid = _selectedId!;
                // 서버에 선택 통지
                try {
                  await _service.notifyStoreSelected(sid);
                } catch (_) {
                  // 실패해도 화면 이동은 진행
                }

                if (!mounted) return; // ★ await 직후 mounted 체크

                final Store selected =
                    _stores.firstWhere((e) => e.id == sid, orElse: () => _stores.first);

                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => ShoppingListScreen(store: selected),
                  ),
                );
              },
              child:
                  const Text('Shoppingo', style: TextStyle(color: Colors.white)),
            ),
          ),
        ],
      ),
    );
  }

    //  매장 썸네일(이미지) 표시
  Widget _storeThumb(String? url) {
    final fixed = _fixImageUrl(url);
    if (fixed != null && fixed.isNotEmpty) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Image.network(
          fixed,
          width: 40,
          height: 40,
          fit: BoxFit.cover,
          errorBuilder: (_, __, ___) => _fallbackStoreIcon(),
        ),
      );
    }
    return _fallbackStoreIcon();
  }

  //  이미지 로딩 실패 또는 URL 없음 → 기본 매장 아이콘
  Widget _fallbackStoreIcon() => Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: const Color(0xFFEDEFF5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: const Icon(Icons.store, color: Colors.black54),
      );

  //  에뮬레이터에서 127.0.0.1 / localhost를 10.0.2.2로 보정
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


  Widget _searchField() => TextField(
        decoration: InputDecoration(
          hintText: '매장 검색',
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
        onChanged: (v) => setState(() => _searchQuery = v),
      );
}
