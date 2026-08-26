# 2025 모용 형태 정량화 연구 프로그램

이 저장소는 2025년 모용(trichome) 연구에서 개발·사용한 Python 프로그램 여섯 개의 **역사적 최종본**을 공개하기 위한 자료입니다. SEM 이미지에서 모용을 분할하고 형태지표를 추출한 뒤, 탐색적 통계·PCA·계층군집으로 이어지는 연구 흐름을 코드로 연결했습니다.

> 공개일: 2026-08-26  
> 코드 시점: 2025년 연구 당시 최종 사용본  
> 상태: 연구용 프로토타입·역사적 스냅샷

`src/`의 여섯 파일은 보관 중인 최종 코드와 SHA-256이 일치하도록 그대로 복사했습니다. 2026년 공개 준비 과정에서 확인한 오류를 원본 코드에 소급해 고치지 않았으며, 자세한 내용은 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)에 적었습니다.

## 1분 요약

| 질문 | 답 |
|---|---|
| 무엇을 해결하려 했나? | 육안 중심의 모용 형태 관찰을 이미지 기반 수치형 특징으로 바꾸고, 종·과 사이의 형태 유사성을 탐색하려 했습니다. |
| 무엇을 만들었나? | 주석·분할·골격화·형태지표 추출, CSV 병합, 분포 진단, 탐색적 통계, PCA·Ward 군집, 이진 성분 유사도 군집 프로그램입니다. |
| 개인 역할은? | 연구 참여자의 진술에 따르면 이 저장소의 코딩과 통계 분석은 이현신이 담당했습니다. 연구 전체는 팀 연구였으며, 코드만으로 개인 기여도를 독립 증명하지는 않습니다. |
| 가장 큰 가치가 무엇인가? | 생물학적 관찰을 측정 가능한 데이터 흐름으로 바꾸고, 분석 가정과 실패를 확인하며 도구를 확장·축소한 개발 과정입니다. |
| 무엇을 주장하지 않나? | 검증된 자동 분류기, 일반 목적 분석 패키지, 형태·성분 계통수, 또는 원 논문 결과의 완전한 독립 재현을 주장하지 않습니다. |

## 프로그램 구성

| 파일 | 역할 | 핵심 입출력 |
|---|---|---|
| [`trichome_pipeline.py`](src/trichome_pipeline.py) | 이미지 주석·분할, 골격화, 형태지표 추출, QC | 이미지·instance mask → `features_summary.csv`, mask, QC PNG |
| [`merge_and_analyze.py`](src/merge_and_analyze.py) | 종별 CSV 병합, one-way ANOVA, Tukey HSD | 종별 CSV 폴더 → 병합 CSV, 분석 로그 |
| [`normality_validation.py`](src/normality_validation.py) | 종×형질별 Shapiro-Wilk 진단과 분포 시각화 | 병합 CSV → stdout, 다중 페이지 PDF |
| [`advanced_statistical_analysis.py`](src/advanced_statistical_analysis.py) | ANOVA, 변환 후 ANOVA, Kruskal-Wallis의 탐색적 비교 | 병합 CSV → stdout |
| [`advanced_trichome_analyzer.py`](src/advanced_trichome_analyzer.py) | 표준화 PCA, 시각화, 종 평균 Ward 군집, bootstrap, Newick | 종별 CSV 폴더 → CSV·TXT·PNG·NWK |
| [`component_tree_builder_v5.py`](src/component_tree_builder_v5.py) | 이진 성분자료의 SMC/Jaccard UPGMA 유사도 군집 | 성분×종 CSV → PNG·NWK |

Ward·UPGMA 산출물은 **형태 또는 성분의 유사도 군집도(phenogram)**이며, 진화계통수의 증거로 해석하지 않습니다.

## 저장소 구조

```text
.
├─ README.md
├─ PROVENANCE.md
├─ CONTRIBUTIONS.md
├─ DEVELOPMENT_HISTORY.md
├─ KNOWN_LIMITATIONS.md
├─ VALIDATION.md
├─ requirements.txt
├─ SHA256SUMS.txt
├─ src/                         # 2025 최종 코드 6개: 수정하지 않은 보존본
└─ examples/                    # 실제 연구자료가 아닌 합성 실행 예제
   ├─ morphometrics/
   └─ components/
```

원본 SEM 이미지, 마스크, 연구팀 자료와 실제 결과 CSV는 개인정보·공동연구 자료·용량 문제 때문에 이 공개본에 포함하지 않았습니다. 따라서 이 저장소는 코드 구조와 실행 인터페이스를 공개하지만, 2025년 결과 전체의 독립 재현 패키지는 아닙니다.

## 설치

소스에는 2025년 당시의 정확한 Python·패키지 버전 기록이 남아 있지 않습니다. 아래 파일은 필요한 패키지 이름만 제시하며, 역사적 환경을 소급해 주장하지 않습니다.

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

아래 예제 CSV는 프로그램 입출력 형식 확인을 위해 2026년에 만든 **합성자료**입니다. 연구 결과나 실제 측정값이 아닙니다.

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

`--mode individual`은 도움말과 달리 개체 수준 덴드로그램을 만들지 않고 PCA까지만 실행합니다. 보존본의 실제 동작을 숨기지 않기 위해 그대로 두었습니다.

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

## 읽을 때의 기준

- [`src/`](src/): 2025년 코드 보존본
- [`examples/`](examples/): 2026년 공개 준비를 위해 만든 합성자료
- [`PROVENANCE.md`](PROVENANCE.md): 파일 선별·해시·시점의 경계
- [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md): 팀 연구와 개인 기여의 경계
- [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md): 중간 코드·산출물에서 재구성한 시도–개선–실패–롤백 흐름
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md): 2026년 사후 감사에서 확인한 구현·방법론 한계
- [`VALIDATION.md`](VALIDATION.md): 공개본에서 실제로 확인한 검증 범위

GitHub 게시일이나 이후 commit은 2025년 개발 시점을 증명하지 않습니다. 당시 연구일지, 원본 압축파일, 산출물과 수상 자료가 별도의 출처 증거이며, 이 저장소는 그중 코드 부분을 읽기 쉽게 공개한 보조 자료입니다.

## 이용 조건

현재 별도의 오픈소스 라이선스를 부여하지 않았습니다. 공개 열람은 가능하지만, 명시적인 허가 없이 복제·수정·재배포할 권리를 부여한다는 뜻은 아닙니다. 공동연구·학교 관련 권리를 확인한 뒤 라이선스를 별도로 정할 수 있습니다.
