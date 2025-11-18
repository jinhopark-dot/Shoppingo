//lib\models\route_result.dart
import 'dart:convert';

class RouteResult {
  final List<String> orderedNode;

  /// 노드별 상품 ID 묶음 (2차원). 예: [[120,121],[130],[]]
  final List<List<int>> orderedProductIds;

  /// 서버가 준 이미지 URL(상대/절대 가능)
  final String routeImageUrl;

  const RouteResult({
    required this.orderedNode,
    required this.orderedProductIds,
    required this.routeImageUrl,
  });

  factory RouteResult.fromJson(Map<String, dynamic> j) {
    // --- ordered_node: ["N1","N2"] 또는 문자열 "[\"N1\",\"N2\"]" 모두 처리
    List<String> parseNodes(dynamic v) {
      if (v is List) return v.map((e) => e.toString()).toList();
      if (v is String) {
        try {
          final dec = jsonDecode(v);
          if (dec is List) return dec.map((e) => e.toString()).toList();
        } catch (_) {}
        final s = v.trim();
        final core = s.startsWith('[') && s.endsWith(']') ? s.substring(1, s.length - 1) : s;
        if (core.trim().isEmpty) return <String>[];
        return core.split(',').map((e) => e.trim()).toList();
      }
      return <String>[];
    }

    // --- ordered_product_ids: 어떤 혼종이 와도 2차원 정수 리스트로 정규화 ---
    List<List<int>> parseProductGroups(dynamic v) {
      // 그룹 하나(g)에서 "정수"만 뽑아내는 헬퍼
      List<int> extractInts(dynamic g) {
        // 1) 숫자 하나
        if (g is num) return <int>[g.toInt()];

        // 2) 배열: [120, "121", 130]
        if (g is List) {
          final out = <int>[];
          for (final x in g) {
            if (x is num) {
              out.add(x.toInt());
            } else {
              final m = RegExp(r'-?\d+').firstMatch(x.toString());
              if (m != null) out.add(int.parse(m.group(0)!));
            }
          }
          return out;
        }

        // 3) 문자열: "[118]" / "[120, 121]" / "118" / "상품:118"
        if (g is String) {
          final matches = RegExp(r'-?\d+').allMatches(g);
          return [for (final m in matches) int.parse(m.group(0)!)];
        }

        // 그 외는 빈 리스트
        return <int>[];
      }

      // 최상위가 리스트인 보통 케이스: [ "[118]", [120,121], 130, ... ]
      if (v is List) {
        return [for (final g in v) extractInts(g)];
      }

      // 최상위가 문자열인 케이스: "[[118],[120,121]]" 또는 "[118,119]"
      if (v is String) {
        // 먼저 JSON 시도
        try {
          final dec = jsonDecode(v);
          return parseProductGroups(dec);
        } catch (_) {
          // JSON 아니면 문자열 전체에서 숫자만 찾아 한 묶음으로 처리
          final row = extractInts(v);
          return <List<int>>[row];
        }
      }

      return <List<int>>[];
    }


    final nodes = parseNodes(j['ordered_node']);
    final groups = parseProductGroups(j['ordered_product_ids']);

    final url = j['route_image_url'];
    if (url is! String || url.isEmpty) {
      throw const FormatException('route_image_url 누락');
    }

    return RouteResult(
      orderedNode: nodes,
      orderedProductIds: groups,
      routeImageUrl: url,
    );
  }

  /// 필요하면 화면 표시용으로 평탄화한 리스트가 필요할 때 사용
  List<int> get flatProductIds => [
        for (final g in orderedProductIds) ...g,
      ];
}

