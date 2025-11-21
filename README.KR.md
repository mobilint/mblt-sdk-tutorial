# Mobilint SDK 튜토리얼

<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="./assets/Mobilint_Logo.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

이 레포지토리는 컴파일러(qubee)와 런타임 소프트웨어(maccel) 라이브러리를 포함하는 Mobilint SDK qb를 쉽게 시작할 수 있도록 예제와 설명을 제공합니다.

컴파일러를 사용하여 변환된 모델은 런타임을 통해 Mobilint NPU에서 실행할 수 있습니다. 적절히 구성되면, 이 워크플로우를 통해 모델이 원본 모델의 정확도를 유지하면서 더 빠른 추론 성능을 달성할 수 있습니다.

## 시작하기 전에

시작하기 전에 Mobilint NPU를 가지고 계시는지 확인하세요.
NPU를 가지고 계시지 않다면, AI 애플리케이션 평가 옵션에 대해 논의하기 위해 [문의](mailto:contact@mobilint.com)바랍니다.

SDK는 [Mobilint 다운로드 센터](https://dl.mobilint.com/)를 통해 배포됩니다. SDK를 다운로드하기 전에 계정에 가입하고 권한을 얻어야 합니다.

## 개요

<div align="center">
<img src="./assets/Compiler.avif" width="45%", alt="Compiler Diagram">
<img src="./assets/Runtime.avif" width="45%", alt="Runtime Diagram">
</div>

Mobilint [SDK qb](https://www.mobilint.com/sdk-qb)는 [컴파일러](compilation/README.md)와 [런타임](runtime/README.md) 두 가지 주요 구성 요소로 구성됩니다.

Mobilint qb 컴파일러는 널리 사용되는 딥러닝 프레임워크의 모델을 Mobilint Model eXeCUtable (MXQ) 형식으로 변환합니다.
사전 훈련된 모델과 캘리브레이션 데이터셋을 사용하여 컴파일러는 모델을 분석하고, 양자화하며, Mobilint NPU에서 실행하기 위해 최적화합니다.

Mobilint qb 런타임은 컴파일된 MXQ 모델을 Mobilint NPU에서 실행할 수 있게 합니다.
런타임 라이브러리를 사용하면 컴파일된 MXQ 모델을 간단하고 효율적인 방식으로 실제 애플리케이션에 통합할 수 있습니다.

자세한 내용은 [컴파일러](compilation/README.md) 및 [런타임](runtime/README.md) 튜토리얼을 참조하세요.

## 지원 및 이슈

이 튜토리얼을 따라하는 동안 문제가 발생하면 [기술 지원 이메일](mailto:tech-support@mobilint.com)로 문의바랍니다.
