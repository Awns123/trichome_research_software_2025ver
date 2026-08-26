# 공개본 검증 기록

## 검증 범위

- 검증일: 2026-08-26
- 운영체제: Windows
- Python: 3.12.13
- 대상: `src/`의 Python 파일 6개와 `examples/`의 합성자료
- 환경변수: `PYTHONUTF8=1`, `MPLBACKEND=Agg`

이 검증은 공개 저장소의 파일 무결성과 기본 실행 경로를 확인한 것입니다. 2025년 당시 환경을 복원하거나 실제 연구 결과의 과학적 타당성을 재검증한 것은 아닙니다.

## 사용한 직접 의존성 버전

```text
numpy==2.5.2
pandas==3.0.5
scipy==1.18.1
matplotlib==3.11.1
seaborn==0.13.2
scikit-learn==1.9.0
statsmodels==0.14.6
opencv-python==5.0.0.93
pillow==12.3.0
```

`requirements.txt`는 2025년의 정확한 환경이 남아 있지 않아 패키지 이름만 기록합니다. 위 버전 목록은 2026-08-26 검증 환경일 뿐 역사적 환경이라고 주장하지 않습니다.

## 통과한 검사

| 검사 | 결과 |
|---|---|
| Python AST 구문 분석 | 6/6 통과 |
| `--help` 호출 | 6/6 통과 (`PYTHONUTF8=1`) |
| 여섯 공개 코드와 선별 원본의 SHA-256 대조 | 6/6 일치 |
| 합성 원영상+instance mask로 `trichome_pipeline.py` 실행 | 종료코드 0, 1행 CSV와 QC PNG 생성 |
| 종별 합성 CSV 병합·ANOVA 로그 생성 | 종료코드 0 |
| 합성 병합 CSV의 정규성 PDF 생성 | 종료코드 0 |
| 합성 병합 CSV의 탐색적 고급 통계 실행 | 종료코드 0 |
| 합성 형태자료의 PCA·Ward·Newick 생성 | 종료코드 0 |
| 합성 이진 성분자료의 SMC/Jaccard UPGMA·Newick 생성 | 종료코드 0 |

테스트 결과물은 32개가 생성되었으나 `results/`는 `.gitignore`로 공개본에서 제외합니다.

## 관찰된 경고와 실패 경계

- headless `Agg` backend에서는 원 코드의 `plt.show()` 호출에 비대화형 경고가 나지만 파일 저장은 완료되었습니다.
- 작은 합성자료의 일부 dendrogram에서 `tight_layout` 경고가 났지만 PNG·NWK는 생성되었습니다.
- 처음 만든 합성자료에서 `num_endpoints`와 `num_branchpoints`가 종마다 상수였을 때 `normality_validation.py`의 KDE가 singular covariance 오류로 중단되었습니다. 예제값에 변이를 주어 전체 경로를 확인했으며, 이 취약점은 `KNOWN_LIMITATIONS.md`에 기록했습니다.
- CP949 콘솔에서 `trichome_pipeline.py --help`는 `µ` 문자 때문에 `UnicodeEncodeError`가 났습니다. `PYTHONUTF8=1` 설정 후 통과했으며 README에 Windows 설정을 추가했습니다.

