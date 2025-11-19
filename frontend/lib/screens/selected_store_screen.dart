// lib/screens/selected_store_screen.dart
import 'package:flutter/material.dart';
import '../models/store.dart';

class SelectedStoreScreen extends StatelessWidget {
  final Store store;
  const SelectedStoreScreen({super.key, required this.store});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(store.name)),
      body: Center(
        child: Text('선택된 매장 ID: ${store.id}'),
      ),
    );
  }
}
