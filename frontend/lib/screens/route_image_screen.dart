import 'package:flutter/material.dart';
import '../models/route_result.dart';

class RouteImageScreenUrl extends StatelessWidget {
  final RouteResult result;
  const RouteImageScreenUrl({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('경로 안내')),
      body: InteractiveViewer(
        minScale: 0.5,
        maxScale: 5,
        child: Center(
          child: Image.network(
            result.routeImageUrl,
            fit: BoxFit.contain,
            errorBuilder: (_, err, __) =>
                Padding(padding: const EdgeInsets.all(24), child: Text('이미지를 불러오지 못했습니다.\n$err')),
          ),
        ),
      ),
      bottomNavigationBar: (result.orderedNode.isNotEmpty)
          ? Padding(
              padding: const EdgeInsets.all(12),
              child: Text(
                '경유 노드 수: ${result.orderedNode.length}',
                textAlign: TextAlign.center,
              ),
            )
          : null,
    );
  }
}
