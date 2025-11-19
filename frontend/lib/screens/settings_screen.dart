// lib/screens/settings_screen.dart
import 'package:flutter/material.dart';
import '../data/app_config.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _form = GlobalKey<FormState>();
  late final TextEditingController _url;

  @override
  void initState() {
    super.initState();
    _url = TextEditingController(text: AppConfig.I.baseUrl);
  }

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  String? _validateUrl(String? v) {
    final s = (v ?? '').trim();
    if (s.isEmpty) return 'URL을 입력하세요.';
    // 단순 검증: http/https 시작, 공백 없음
    if (!(s.startsWith('http://') || s.startsWith('https://'))) {
      return 'http:// 또는 https:// 로 시작해야 합니다.';
    }
    // ngrok은 https 권장(인증서 문제 회피). http를 쓸 땐 안드로이드 cleartext 허용필요.
    return null;
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    await AppConfig.I.setBaseUrl(_url.text);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('서버 URL이 저장되었습니다.')),
    );
    Navigator.of(context).pop(true); // 저장 후 뒤로
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('설정')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _form,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('서버 URL 입력', style: TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 8),
              TextFormField(
                controller: _url,
                decoration: const InputDecoration(
                  hintText: '예) https://1234-56-78-90.ngrok-free.app',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.url,
                validator: _validateUrl,
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _save,
                      child: const Text('저장'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const Text(
                'ngrok 주소는 8시간마다 갱신됩니다. 갱신 시 이 화면에서 새 URL을 다시 저장하십시오.',
                style: TextStyle(color: Colors.black54),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
