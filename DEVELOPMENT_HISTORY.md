# 개발 과정

이 문서는 2025년 연구 아카이브에 남아 있는 코드와 산출물을 대조해 정리한 프로그램 개발 흐름입니다. Git 기록을 대신하는 문서가 아니며, 공개 저장소에는 연구에 사용한 최종 여섯 파일만 수록했습니다.

## 1. 이미지에서 형태지표까지

`trichome_pipeline.py`는 모용 이미지 처리의 여러 단계를 한 프로그램으로 연결했습니다.

1. 수동 다각형 주석
2. JPEG·PNG·다중 페이지 TIFF 입력
3. 스케일바 두 점 클릭을 통한 화소 크기 보정
4. instance mask 생성·재사용
5. 골격화와 endpoint/branch 후보 탐색
6. 중심선·폭·곡률·기본 형상 지표 추출
7. flood-fill 자동 주석
8. polygon ROI 안의 Otsu·GrabCut 분할
9. tip/root 클릭을 사용한 dual-curve 경계 보정
10. 재귀 폴더 처리, 이미지 해시 중복 억제, QC·통합 CSV 저장

연구 과정에서 이미지 입력부터 정량지표 CSV 생성까지 반복 작업을 하나의 흐름으로 묶은 최종 통합 프로그램입니다.

## 2. 형태 PCA·Ward 분석

| 단계 | 주요 변화 |
|---|---|
| V0 | CSV 병합, 표준화, PCA, Ward 군집의 기본 흐름 구현 |
| V1 | 식별 열 제외, 데이터 점검, 산출물 확대 |
| V1-BW | 흑백 발표용 시각화 추가 |
| V2 | 과 매핑과 과별 PCA 그림 추가 |
| V3 | 과별 dendrogram과 Newick 저장 추가 |
| V4 | scree, loading, feature bootstrap 추가 |
| V4-log | 로그변환 기능을 시험한 중간 코드 |
| V5 | PCA 좌표·설명력 저장, global/zoom/biplot, 범례 분리 |

공개한 `advanced_trichome_analyzer.py`는 V5입니다. 분석 결과를 CSV와 그림으로 남기고, 전체·확대·biplot 시각화를 비교할 수 있게 확장한 최종 사용본입니다.

## 3. 통계 분석 도구의 확장

2025-10-07 아카이브에는 다음 세 도구가 순차적으로 남아 있습니다.

| 시각(KST) | 파일 | 추가한 기능 |
|---|---|---|
| 15:44 | `merge_and_analyze.py` | 종별 CSV 병합과 ANOVA·Tukey 자동화 |
| 16:37 | `advanced_statistical_analysis.py` | 변환 ANOVA와 Kruskal-Wallis 비교 |
| 17:02 | `normality_validation.py` | 종×형질별 Shapiro 결과와 분포 그림 기록 |

하나의 검정 결과에 머물지 않고 분포를 확인하고 모수·비모수 분석을 비교할 수 있도록 기능을 확장했습니다.

## 4. 이진 성분 유사도 군집

| 단계 | 주요 변화 |
|---|---|
| C1 | SMC·Jaccard 거리, UPGMA, feature bootstrap 구현 |
| C2 | Dice와 Neighbor-Joining 기능 시험 |
| C3 | 거리행렬 변환 함수 분리 |
| C4/V5 | SMC·Jaccard UPGMA와 전체·과별 분석으로 구성 정리 |

공개한 `component_tree_builder_v5.py`는 C4/V5입니다. 성분의 존재·부재 행렬을 읽어 SMC와 Jaccard 거리 기반의 전체·과별 유사도 군집도와 Newick 파일을 생성합니다.

## 개발 흐름 요약

```text
정성 관찰
  → 이미지 주석·분할
  → 형태지표 정량화
  → 종별 자료 병합과 분포 확인
  → PCA·통계·계층군집
  → 결과표·그림·Newick 저장
```

여섯 프로그램은 생물학적 관찰을 반복 가능한 데이터 처리와 분석 단계로 연결하려 한 2025년 연구 개발의 결과물입니다.

