# 공개본 실행 확인 기록

## 확인 범위

- 확인일: 2026-08-26
- 운영체제: Windows
- Python: 3.12.13
- 대상: `src/`의 Python 파일 6개와 `examples/`의 합성자료
- 환경변수: `PYTHONUTF8=1`, `MPLBACKEND=Agg`

이 기록은 공개 저장소에서 프로그램의 구문, 명령행 진입점과 합성자료 실행 경로를 확인한 결과입니다. 실제 연구자료는 공개본에 포함하지 않았습니다.

## 확인 환경의 직접 의존성 버전

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

`requirements.txt`는 패키지 이름을 기록합니다. 위 목록은 2026-08-26 공개본 실행 확인에 사용한 환경입니다.

## 확인 결과

| 항목 | 결과 |
|---|---|
| Python AST 구문 분석 | 6/6 통과 |
| `--help` 호출 | 6/6 통과 (`PYTHONUTF8=1`) |
| 여섯 공개 코드와 선별 원본의 SHA-256 대조 | 6/6 일치 |
| 합성 원영상+instance mask로 `trichome_pipeline.py` 실행 | 종료코드 0, 1행 CSV와 QC PNG 생성 |
| 종별 합성 CSV 병합·ANOVA 로그 생성 | 종료코드 0 |
| 합성 병합 CSV의 정규성 PDF 생성 | 종료코드 0 |
| 합성 병합 CSV의 탐색적 통계 실행 | 종료코드 0 |
| 합성 형태자료의 PCA·Ward·Newick 생성 | 종료코드 0 |
| 합성 이진 성분자료의 SMC/Jaccard UPGMA·Newick 생성 | 종료코드 0 |

실행 중 생성되는 `results/` 폴더는 `.gitignore`에 따라 저장소에서 제외합니다.

## 실행 참고

- GUI가 필요한 주석·보정 기능은 대화형 Matplotlib backend에서 사용합니다.
- headless `Agg` backend에서는 `plt.show()` 관련 비대화형 경고가 표시될 수 있으나 파일 저장은 완료됩니다.
- 일부 한국어 Windows 콘솔에서는 `µ` 문자 출력 때문에 인코딩 오류가 날 수 있어 `PYTHONUTF8=1` 설정을 권장합니다.
- 합성 예제는 입력 형식과 실행 경로 확인용이며 실제 연구 측정값이 아닙니다.
