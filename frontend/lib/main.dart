import 'dart:async';
import 'package:flutter/material.dart';
import 'data/app_config.dart';
import 'constants.dart';
import 'screens/store_select_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Flutter 프레임워크 에러도 콘솔로
  FlutterError.onError = (FlutterErrorDetails details) {
    FlutterError.dumpErrorToConsole(details);
  };

  // 모든 top-level 예외 로그로 잡기
  runZonedGuarded(() async {
    // 여기서 SharedPreferences 같은 플러그인 호출 가능
    await AppConfig.I.init();

    // 선택: 기본 URL이 있으면 초기 세팅
    if (!AppConfig.I.hasUrl && kDefaultBaseUrl.isNotEmpty) {
      await AppConfig.I.setBaseUrl(kDefaultBaseUrl);
    }

    runApp(const MyApp());
  }, (Object error, StackTrace stack) {
    // 콘솔에 정확한 원인 출력
    // 무조건 보이게 print
    // ignore: avoid_print
    print('FATAL during main(): $error\n$stack');
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'shoppingo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: false,
        scaffoldBackgroundColor: const Color(0xFFF5F6F8),
        appBarTheme: const AppBarTheme(
          backgroundColor: kAppBlue,
          foregroundColor: Colors.white,
          centerTitle: true,
          iconTheme: IconThemeData(color: Colors.white),
          actionsIconTheme: IconThemeData(color: Colors.white),
          titleTextStyle: TextStyle(
            color: Colors.white, fontSize: 20, fontWeight: FontWeight.w600),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ButtonStyle(
            foregroundColor: WidgetStatePropertyAll(Colors.white),
            backgroundColor: WidgetStatePropertyAll(kAppBlue),
          ),
        ),
      ),
      home: const StoreSelectScreen(),
    );
  }
}

