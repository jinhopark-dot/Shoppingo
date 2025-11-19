// lib/services/store_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../data/app_config.dart';
import '../models/store.dart';
import '../models/product.dart';
import '../models/route_result.dart';

class StoreService {
  final http.Client _client;
  StoreService({http.Client? client}) : _client = client ?? http.Client();

  String get _baseUrl => AppConfig.I.baseUrl;

  // 상대 경로를 절대 URL로 보정
  String _absUrl(String maybePath) {
    // _fixImageUrl가 null도 받으니 그대로 위임
    return _fixImageUrl(maybePath) ?? maybePath;
  }

  // 1) 매장 목록
  Future<List<Store>> fetchStores() async {
    if (_baseUrl.isEmpty) throw Exception('서버 URL이 비어 있습니다.');

    final resp = await _client.get(Uri.parse('$_baseUrl/api/stores'));
    if (resp.statusCode != 200) {
      throw Exception('매장 조회 실패: ${resp.statusCode} ${resp.body}');
    }

    final decoded = jsonDecode(resp.body);
    if (decoded is! List) {
      throw Exception('매장 응답이 배열이 아님: ${decoded.runtimeType}');
    }

    // JSON → Store 변환 + 이미지 URL 보정
    final stores = (decoded)
        .map((e) => Store.fromJson(e as Map<String, dynamic>))
        .map((s) => Store(
              id: s.id,
              name: s.name,
              address: s.address,
              imageUrl: _fixImageUrl(s.imageUrl),
            ))
        .toList();

    return stores;
  }


  // 2) 특정 매장의 상품 목록
  Future<List<Product>> fetchInventory(int storeId) async {
    if (_baseUrl.isEmpty) throw Exception('서버 URL이 비어 있습니다.');
    final resp = await _client.get(Uri.parse('$_baseUrl/api/stores/$storeId/products'));
    if (resp.statusCode != 200) {
      throw Exception('재고 조회 실패: ${resp.statusCode} ${resp.body}');
    }
    final decoded = jsonDecode(resp.body);
    if (decoded is! List) {
      throw Exception('재고 응답이 배열이 아님: ${decoded.runtimeType}');
    }
    return decoded.map((e) => Product.fromJson(e as Map<String, dynamic>)).toList();
  }

  // 3) 선택한 매장 서버에 통지(백엔드가 안 쓰면 무시해도 됨)
  Future<void> notifyStoreSelected(int storeId) async {
    if (_baseUrl.isEmpty) return;
    final uri = Uri.parse('$_baseUrl/api/selected-store');
    try {
      final resp = await _client.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'store_id': storeId}),
      );
      // 백엔드에 이 엔드포인트 없으면 404 나올 수 있음. 무시 가능.
      if (resp.statusCode >= 400 && resp.statusCode != 404) {
        // 필요하면 로그만 남기고 넘어가도 됨.
      }
    } catch (_) {
      // 오프라인/엔드포인트 미구현 등은 프론트 기능에 치명적이지 않으니 무시.
    }
  }

  // 4) 최적 경로 생성 (URL 응답 버전)
  Future<RouteResult> createRouteResult({
    required int storeId,
    required List<int> productIds,
    required String startNode,
  }) async {
    if (_baseUrl.isEmpty) {
      throw Exception('서버 URL이 비어 있습니다.');
    }
    final payload = {
      'store_id': storeId,
      'product_ids': productIds,
      'start_node': startNode,
    };

    final uri = Uri.parse('$_baseUrl/api/routes');
    final resp = await _client.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    );

    final ct = resp.headers['content-type'] ?? '';
    if (resp.statusCode != 200) {
      // 백엔드가 422 등을 주면 여기로 들어옵니다.
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
    if (!ct.contains('application/json')) {
      final preview = resp.body.length > 120 ? resp.body.substring(0, 120) : resp.body;
      throw Exception('JSON 응답이 아님: $ct / "$preview"');
    }

    final decoded = jsonDecode(resp.body);
    if (decoded is! Map<String, dynamic>) {
      throw Exception('예상치 못한 JSON 형태: ${decoded.runtimeType}');
    }

    final result = RouteResult.fromJson(decoded);
    return RouteResult(
      orderedNode: result.orderedNode,
      orderedProductIds: result.orderedProductIds,
      routeImageUrl: _absUrl(result.routeImageUrl),
    );
    
  }

  String? _fixImageUrl(String? raw) {
  if (raw == null || raw.isEmpty) return null;

  var value = raw;

  // localhost/127.0.0.1 → 에뮬레이터에서 PC 접근용 10.0.2.2
  if (value.startsWith('http://127.0.0.1') ||
      value.startsWith('http://localhost')) {
    value = value.replaceFirst(
      RegExp(r'http://(127\.0\.0\.1|localhost)'),
      'http://10.0.2.2',
    );
  }

  // 상대 경로면 baseUrl 붙이기
  if (value.startsWith('/')) {
    final base = _baseUrl.endsWith('/')
        ? _baseUrl.substring(0, _baseUrl.length - 1)
        : _baseUrl;
    return '$base$value';
  }

  return value;
}

}

