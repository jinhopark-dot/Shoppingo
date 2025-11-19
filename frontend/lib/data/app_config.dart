// lib/data/app_config.dart
import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  AppConfig._();
  static final AppConfig I = AppConfig._();

  static const _kBaseUrlKey = 'base_url';

  String _baseUrl = ''; // 메모리 캐시

  String get baseUrl => _baseUrl;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString(_kBaseUrlKey) ?? '';
  }

  Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = url.trim();
    await prefs.setString(_kBaseUrlKey, _baseUrl);
  }

  bool get hasUrl => _baseUrl.isNotEmpty;
}
