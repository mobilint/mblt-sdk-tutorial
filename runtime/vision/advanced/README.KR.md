# 이미지 분류 모델을 이용한 심화 테스트 (Advanced Test with Image Classification Model)

이 튜토리얼은 모빌린트(Mobilint) qb 런타임을 사용하여 컴파일된 이미지 분류 모델로 인퍼런스를 실행하는 방법에 대한 자세한 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/vision/advanced/README.md`에서 이어집니다. 모델 컴파일을 성공적으로 마쳤으며 다음 파일들이 준비되어 있다고 가정합니다.

- `ILSVRC2012_bbox_val_v3.tgz`
- `ILSVRC2012_img_val.tar`

다음 명령어를 사용하여 이 파일들을 가져올 수 있습니다.

```bash
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

## 사전 요구 사항 (Prerequisites)

인퍼런스를 실행하기 전에 다음 사항을 확인하십시오.

- maccel 런타임 라이브러리 (NPU 가속기 액세스 제공)
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `PIL`, `numpy`, `torch`

그런 다음 `mblt-model-zoo` 패키지의 벤치마크 스크립트를 사용하여 인퍼런스를 실행합니다.

따라서 저장소를 복제하고 패키지를 설치합니다.

```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
git checkout jm/vision_bench # `jm/vision_bench` 브랜치로 전환
pip install -e.
cd tests/vision/benchmark
```

## 데이터셋 정리 (Organize Dataset)

벤치마크를 사용하려면 먼저 데이터셋을 정리해야 합니다. 다음 명령어는 벤치마크 스크립트가 예상하는 디렉토리 구조를 생성합니다.

```bash
python organize_imagenet.py --image_dir ../../../../ILSVRC2012_img_val.tar --xml_dir ../../../../ILSVRC2012_bbox_val_v3.tgz
```

위의 명령어를 실행한 후, 데이터셋은 `~/.mblt_model_zoo/datasets/imagenet` 디렉토리에 정리됩니다.

## 벤치마크 실행 (Run Benchmark)

벤치마크를 실행하려면 다음 명령어를 사용하십시오.

```bash
python benchmark_imagenet.py --local_path {path_to_local_mxq_file} --model_type {model_configuration_type} --infer_mode {core_allocation_mode} --batch_size {batch_size} 
```

예를 들어, 명령어는 다음과 같습니다.

```bash
python benchmark_imagenet.py --local_path /workspace/mblt-sdk-tutorial/compilation/vision/advanced/resnet50_5cls_100_9999_01.mxq --model_type IMAGENET1K_V1 --infer_mode single --batch_size 1 
```
