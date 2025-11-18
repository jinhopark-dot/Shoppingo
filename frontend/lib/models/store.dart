class Store {
  final int id;
  final String name;
  final String? address;
  final String? imageUrl;

  Store({
    required this.id,
    required this.name,
    this.address,
    this.imageUrl,
  });

  factory Store.fromJson(Map<String, dynamic> json) {
    return Store(
      id: json['id'] as int,
      name: (json['name'] ?? '') as String,
      address: json['address'] as String?,
      imageUrl: json['image_url'] as String?,   // ← 서버 키와 동일
    );
  }
}
