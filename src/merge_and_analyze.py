import pandas as pd
import numpy as np
import os
import argparse
import sys  # 로그 저장을 위해 sys 모듈 추가
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# (추가) 터미널 출력과 파일 저장을 동시에 수행하는 Logger 클래스
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def merge_and_analyze(input_dir, output_csv):
    """
    지정된 폴더의 모든 CSV 파일을 병합하고, 통계 분석을 수행합니다.
    """
    # --- 1. CSV 파일 병합 ---
    all_dataframes = []
    print(f"'{input_dir}' 폴더에서 CSV 파일 병합을 시작합니다...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            species_name = os.path.splitext(filename)[0]
            try:
                df = pd.read_csv(file_path)
                df['species'] = species_name
                all_dataframes.append(df)
                print(f"  - '{filename}' 파일 처리 완료. (데이터 {len(df)}개)")
            except Exception as e:
                print(f"  - 경고: '{filename}' 파일을 읽는 중 오류 발생: {e}")

    if not all_dataframes:
        print("오류: 처리할 CSV 파일이 없습니다.")
        return

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\n성공! 총 {len(combined_df)}개의 데이터를 병합하여 '{output_csv}' 파일에 저장했습니다.")

    # --- 2. 통계 분석 수행 ---
    print("\n" + "=" * 60)
    print("병합된 데이터를 기반으로 통계적 유의성 검증을 시작합니다.")
    print("=" * 60)

    feature_columns = combined_df.select_dtypes(include=np.number).columns.tolist()
    species_list = combined_df['species'].unique()

    if len(species_list) < 2:
        print("오류: 비교할 종 그룹이 2개 미만입니다. 통계 분석을 중단합니다.")
        return

    for feature in feature_columns:
        print(f"\n--- 특징: '{feature}' 분석 ---")

        feature_data = combined_df[['species', feature]].copy().dropna()
        grouped_data = [feature_data[feature][feature_data['species'] == s] for s in species_list]

        f_statistic, p_value = f_oneway(*grouped_data)

        print(f"분산 분석 (ANOVA) 결과:")
        print(f"  - F-statistic: {f_statistic:.4f}")
        print(f"  - P-value: {p_value:.8f}")

        if p_value < 0.05:
            print("  -> 결론: P-value가 0.05보다 작으므로, 종 그룹 간에 유의미한 차이가 존재합니다.")
            if len(species_list) > 2:
                print("\n   사후 분석 (Tukey's HSD)을 수행합니다...")
                tukey_result = pairwise_tukeyhsd(endog=feature_data[feature],
                                                 groups=feature_data['species'],
                                                 alpha=0.05)

                results_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                          columns=tukey_result._results_table.data[0])

                float_cols = ['meandiff', 'p-adj', 'lower', 'upper']
                for col in float_cols:
                    results_df[col] = pd.to_numeric(results_df[col]).apply(lambda x: f"{x:.8f}")

                print(results_df.to_string(index=False))

        else:
            print("  -> 결론: P-value가 0.05보다 크므로, 이 특징만으로는 유의미한 차이를 단정하기 어렵습니다.")


def main():
    parser = argparse.ArgumentParser(description="여러 종별 CSV 파일을 하나로 병합하고 통계 분석을 수행합니다.")
    parser.add_argument("input_dir", type=str, help="종별 CSV 파일들이 들어있는 폴더 경로")
    parser.add_argument("output_csv", type=str, help="병합된 결과를 저장할 새로운 CSV 파일 경로")
    # (추가) 결과를 저장할 텍스트 파일 경로 인자
    parser.add_argument("output_txt", type=str, help="모든 출력 결과를 저장할 텍스트 파일 경로")
    args = parser.parse_args()

    # (추가) Logger를 설정하여 출력을 파일과 터미널에 동시에 보냅니다.
    sys.stdout = Logger(args.output_txt)

    merge_and_analyze(args.input_dir, args.output_csv)


if __name__ == '__main__':
    main()

