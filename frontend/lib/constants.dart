// lib/constants.dart
// 런타임 설정이 없을 때만 사용하는 백업 기본값.
// 운영/테스트 전환에 필요하면 --dart-define 으로 바꿔 실행할 수 있음.
import 'package:flutter/material.dart';

const kAppBlue = Color(0xFF375BE8);
const String kDefaultBaseUrl = String.fromEnvironment(
  'BASE_URL',
  defaultValue: '', // 기본은 비어 둠. 설정화면에서 입력.
);

// 경로 생성 시작 노드 기본값 (백엔드와 합의된 값)
const String kDefaultStartNode = 'S1';