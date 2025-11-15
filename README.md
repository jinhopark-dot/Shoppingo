# 🛒 ShoppinGO: 오프라인 마트 최적 쇼핑 경로 추천 서비스

**25-2 아주대학교 AI융합캡스톤디자인2 프로젝트**

## 🚀 1. 프로젝트 개요

오프라인 마트 고객을 위한 **최적 쇼핑 경로 안내 서비스**
사용자가 쇼핑 리스트를 입력하면, AI가 매장 구조와 상품 위치를 분석하여 **가장 효율적인 쇼핑 경로를 실시간으로 추천**

## 💡 2. 핵심 기술 및 방법론

### 🧠 AI 기반 경로 탐색 (NCH: Neural Constructive Heuristic)

경로 탐색 문제(Traveling Salesperson Problem, TSP 변형)를 해결하기 위해 **HELD (Heavy Encoder-Light Decoder) 구조의 DNN 모델**을 사용

* **Encoder (Heavy):** 입력된 마트 지도의 구조와 상품 노드 간의 복잡한 관계를 학습, 고차원 Embeddings를 생성

* **Decoder (Light):** Encoder의 Embeddings를 바탕으로, 사용자의 쇼핑 리스트에 기반한 최적의 방문 순서를 Autoregressive 방식으로 구성

### 📈 Multi-Policy & Self Imitation Learning (SIL)

* **Multi-Policy**로 다양한 전략을 동시에 탐색하고, **Self Imitation Learning (SIL)**을 통해 그중 **가장 성공적인 경험**을 집중적으로 모방 학습
* 두 가지 조합을 통해 속도와 최종 해의 품질을 향상
## 🛠️ 3. 기술 스택 (Tech Stack)

| 카테고리 (Category) | 기술 스택 (Stack) | 
| :---: | :---: | 
| **🤖 AI Model Core** | <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/PyG-3C2179?style=for-the-badge&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/NetworkX-3776AB?style=for-the-badge&logo=Python&logoColor=white"/> | 
| **🖥️ Backend & Server** | <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"/> | 
| **💾 Database** | <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/> | 
| **📱 Frontend & UI/UX** | <img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=Flutter&logoColor=white"/> | 
| **🛠️ Development & Tools** | <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/> <img src="https://img.shields.io/badge/VS Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/> | 

## 🧑‍💻 4. 프로젝트 참여자

| 이름 | 역할 | 학과 | 이메일 | 
| :---: | :---: | :---: | :---: | 
| **윤승규** | PM, AI 모델 개발 | 산업공학과 | skyoon1109@ajou.ac.kr | 
| **황규헌** | 프론트엔드 개발, UI/UX | 수학과 | modkiserman@ajou.ac.kr | 
| **김민석** | 백엔드 개발, DB 정의 | 심리학과 | minseok0606@ajou.ac.kr | 
| **박진호** | 백엔드 개발, DB 정의 | 심리학과 | jinhopark@ajou.ac.kr |
