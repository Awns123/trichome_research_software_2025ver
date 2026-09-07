# 2025 모용 형태 정량화 연구 프로그램

이 저장소는 2025년 모용(trichome) 연구에서 이현신이 작성하고 연구에 사용한 Python 프로그램 여섯 개를 공개하기 위한 자료입니다. SEM 이미지의 모용을 분할해 형태지표를 추출하고, 자료 병합·통계 분석·PCA·계층군집까지 이어지는 분석 흐름을 코드로 구현했습니다.

> 공개일: 2026-08-26  
> 코드 시점: 2025년 연구 당시 최종 사용본  
> 성격: 학생 연구용 프로그램의 역사적 스냅샷

`src/`의 여섯 파일은 2025년 연구 아카이브에서 선별한 최종 코드와 SHA-256이 일치하는 보존본입니다.

## 1분 요약

| 질문 | 답 |
|---|---|
| 무엇을 해결하려 했나? | 육안 중심의 모용 관찰을 이미지 기반 수치형 특징으로 바꾸고, 종·과 사이의 형태 및 성분 유사성을 탐색하려 했습니다. |
| 무엇을 만들었나? | 주석·분할·골격화·형태지표 추출, CSV 병합, 분포 진단, 탐색적 통계, PCA·Ward 군집, 이진 성분 유사도 군집 프로그램입니다. |
| 누가 코드를 작성했나? | `src/`에 포함된 여섯 Python 프로그램은 이현신이 작성했습니다. |
| 연구는 개인 연구인가? | 연구 전체와 보고서는 팀 공동 산출물이며, 이 저장소는 그중 이현신이 맡은 코딩·통계 분석 부분을 공개합니다. |
| 어떤 범위의 자료인가? | 코드 구조와 실행 인터페이스를 보여 주는 공개본입니다. 원본 SEM 이미지, 실제 측정자료와 연구 결과 전체는 포함하지 않습니다. |

## 프로그램 구성

| 파일 | 역할 | 핵심 입출력 |
|---|---|---|
| [`trichome_pipeline.py`](src/trichome_pipeline.py) | 이미지 주석·분할, 골격화, 형태지표 추출, QC | 이미지·instance mask → `features_summary.csv`, mask, QC PNG |
| [`merge_and_analyze.py`](src/merge_and_analyze.py) | 종별 CSV 병합, one-way ANOVA, Tukey HSD | 종별 CSV 폴더 → 병합 CSV, 분석 로그 |
| [`normality_validation.py`](src/normality_validation.py) | 종×형질별 Shapiro-Wilk 진단과 분포 시각화 | 병합 CSV → stdout, 다중 페이지 PDF |
| [`advanced_statistical_analysis.py`](src/advanced_statistical_analysis.py) | ANOVA, 변환 후 ANOVA, Kruskal-Wallis의 탐색적 비교 | 병합 CSV → stdout |
| [`advanced_trichome_analyzer.py`](src/advanced_trichome_analyzer.py) | 표준화 PCA, 시각화, 종 평균 Ward 군집, bootstrap, Newick | 종별 CSV 폴더 → CSV·TXT·PNG·NWK |
| [`component_tree_builder_v5.py`](src/component_tree_builder_v5.py) | 이진 성분자료의 SMC/Jaccard UPGMA 유사도 군집 | 성분×종 CSV → PNG·NWK |

Ward·UPGMA 산출물은 형태 또는 성분의 유사도 군집도(phenogram)입니다.

## 저장소 구조

```text
.
├─ README.md
├─ PROVENANCE.md
├─ CONTRIBUTIONS.md
├─ DEVELOPMENT_HISTORY.md
├─ VALIDATION.md
├─ requirements.txt
├─ SHA256SUMS.txt
├─ src/                         # 2025 최종 코드 6개
└─ examples/                    # 실제 연구자료가 아닌 합성 실행 예제
   ├─ morphometrics/
   └─ components/
```

원본 SEM 이미지, 마스크, 연구팀 자료와 실제 결과 CSV는 개인정보·공동연구 자료·용량 문제 때문에 이 공개본에 포함하지 않았습니다.

## 설치

소스에는 2025년 당시의 정확한 Python·패키지 버전 기록이 남아 있지 않습니다. `requirements.txt`는 실행에 필요한 패키지 이름을 제시합니다.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"

# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

의존성은 NumPy, pandas, SciPy, Matplotlib, seaborn, scikit-learn, statsmodels, OpenCV, Pillow입니다. 주석·보정 기능은 Matplotlib GUI가 필요합니다. 일부 한국어 Windows 콘솔에서는 소스의 `µ` 문자를 출력할 때 CP949 인코딩 오류가 날 수 있어 `PYTHONUTF8=1` 설정을 권장합니다.

## 빠른 실행

아래 예제 CSV는 프로그램 입출력 형식을 확인하기 위해 2026년에 만든 합성자료입니다. 연구 결과나 실제 측정값이 아닙니다.

### 1. 종별 형태지표 CSV 병합과 탐색적 통계

```bash
mkdir results
python src/merge_and_analyze.py examples/morphometrics results/all_features.csv results/anova_tukey.txt
python src/normality_validation.py results/all_features.csv results/normality_histograms.pdf
python src/advanced_statistical_analysis.py results/all_features.csv > results/advanced_statistics.txt
```

### 2. PCA와 형태 유사도 군집

```bash
python src/advanced_trichome_analyzer.py examples/morphometrics results/morphometrics --mode species --orient right -n 100
```

### 3. 이진 성분 유사도 군집

```bash
python src/component_tree_builder_v5.py examples/components/component_matrix.csv results/components -n 100
```

### 4. 이미지에서 형태지표 추출

기존 instance mask를 사용할 때의 예입니다. mask는 0을 배경, 서로 다른 양의 정수를 각 모용 instance로 갖는 원영상과 같은 크기의 PNG/TIFF여야 합니다.

```bash
python src/trichome_pipeline.py \
  --image data/image.tif \
  --mask data/image_inst.png \
  --pixel_size_um 0.20 \
  --outdir results/extraction \
  --qc
```

폴더 전체를 처리할 수도 있습니다.

```bash
python src/trichome_pipeline.py \
  --input_dir data/images \
  --pixel_size_um 0.20 \
  --outdir results/extraction \
  --recursive --qc
```

수동 주석, 스케일바 클릭 보정(`--calibrate`), flood-fill(`--auto`), polygon+Otsu/GrabCut(`--auto_polygon`)은 대화형 GUI를 사용합니다. 각 옵션은 `python src/trichome_pipeline.py --help`에서 확인할 수 있습니다.

## 문서 안내

- [`src/`](src/): 2025년 연구에서 사용한 최종 코드
- [`examples/`](examples/): 입력 형식과 실행 경로 확인용 합성자료
- [`PROVENANCE.md`](PROVENANCE.md): 파일 출처와 시점
- [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md): 팀 연구와 개인 역할의 구분
- [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md): 프로그램의 버전별 기능 변화
- [`VALIDATION.md`](VALIDATION.md): 공개본의 구문·실행 확인 기록

GitHub 게시일은 코드의 2025년 작성 시점을 대신하지 않습니다. 당시 연구일지, 원본 압축파일, 발표자료와 수상 자료가 별도의 출처 증거이며, 이 저장소는 그중 코드 부분을 읽기 쉽게 공개한 보조 자료입니다.

## 이용 조건

현재 별도의 오픈소스 라이선스를 부여하지 않았습니다. 공개 열람은 가능하지만, 명시적인 허가 없이 복제·수정·재배포할 권리를 부여한다는 뜻은 아닙니다.
