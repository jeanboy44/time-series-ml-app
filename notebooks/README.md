# 노트북
다양한 분석 모델을 실험하기 위한 코드를 저장합니다.

## 코드 실행
### sample.py
샘플 노트북을 실행합니다. 아래의 코드를 실행하기 위해서는 `README.md`의 가이드를 따라 환경설정을 해야합니다.
```
python notebooks/sample.py
```

### yolov5.py
yolov5 를 사용해서 예측하고 결과를 저장합니다. 코드를 실행하면 실행한 경로 내에 `yolov5s.pt` 모델을 다운로드 하고 예측 결과를 `runs/detect/exp` 아래 저장합니다.
```
python notebooks/yolov5.py
```

### MLFlow UI 실행 방법
`notebooks` 폴더로 이동 후 아래의 명령어를 입력한다.
```
mlflow ui
```

### MLFlow에 실험 이력 생성
- `prophet_mlflow_tutorial.ipynb` 노트북 실행
- mlflow ui 접속(http://127.0.0.1:5000)
