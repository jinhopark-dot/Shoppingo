// lib/models/product.dart
class Product {
  final int id;              // 설계서 변수명: id
  final String name;         // 상품명
  final int? price;          // 가격(원). 서버가 없으면 null일 수 있음
  final String? imageUrl;    // 썸네일 URL(옵션)

  const Product({
    required this.id,
    required this.name,
    this.price,
    this.imageUrl,
  });

  factory Product.fromJson(Map<String, dynamic> j) => Product(
        id: j['id'] as int,
        name: j['name'] as String,
        price: (j['price'] is num) ? (j['price'] as num).toInt() : null,
        imageUrl: j['image_url'] as String?, // 서버에서 키명이 다르면 여기만 바꾸면 됨
      );
}
