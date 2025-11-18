//lib/models/route_image.dart
import 'dart:convert';
import 'dart:typed_data';

class RouteImageData {
  final String imageBase64;   // 순수 Base64
  final String? mime;
  final int? distance;

  const RouteImageData({
    required this.imageBase64,
    this.mime,
    this.distance,
  });

  /// 허용 스키마:
  /// A) { "data_url": "data:image/png;base64,....", "distance": 123 }
  /// B) { "image_base64": "....", "mime": "image/png", "distance": 123 }
  /// C) (혹시를 대비) { "image": "...." } 또는 { "image_data": "...." }
  factory RouteImageData.fromJson(Map<String, dynamic> j) {
    // 1) data_url 우선 처리
    final dynamic dataUrlDyn = j['data_url'];
    if (dataUrlDyn is String && dataUrlDyn.isNotEmpty) {
      final comma = dataUrlDyn.indexOf(',');
      final b64 = comma >= 0 ? dataUrlDyn.substring(comma + 1) : dataUrlDyn;
      final mime = dataUrlDyn.startsWith('data:')
          ? dataUrlDyn.substring(5, dataUrlDyn.indexOf(';'))
          : null;
      return RouteImageData(
        imageBase64: b64,
        mime: mime,
        distance: (j['distance'] as num?)?.toInt(),
      );
    }

    // 2) image_base64
    final dynamic b64Dyn =
        j['image_base64'] ?? j['image'] ?? j['image_data']; // 유연하게
    if (b64Dyn is String && b64Dyn.isNotEmpty) {
      final mime = j['mime'] is String ? j['mime'] as String : null;
      return RouteImageData(
        imageBase64: b64Dyn,
        mime: mime,
        distance: (j['distance'] as num?)?.toInt(),
      );
    }

    // 3) 여전히 없으면 명확히 실패 이유를 던진다
    throw FormatException(
      'RouteImageData: image_base64/data_url 누락. 받은 키: ${j.keys.toList()}',
    );
  }

  Uint8List get bytes => base64Decode(imageBase64);
}
