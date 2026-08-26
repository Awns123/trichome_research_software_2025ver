import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import shapiro, f_oneway, kruskal


def advanced_statistical_analysis(features_df):
    """
    정규분포성을 확인하고, 비정규 데이터는 변환 후 ANOVA를 재실행하며,
    비모수적 검정(Kruskal-Wallis) 결과를 함께 제공합니다.
    """
    # 1. 데이터 준비
    feature_columns = features_df.select_dtypes(include=np.number).columns.tolist()
    species_list = features_df['species'].unique()

    print("=" * 70)
    print("고급 통계 분석을 시작합니다 (정규성 검정, 데이터 변환, 비모수 검정).")
    print("=" * 70)

    for feature in feature_columns:
        print(f"\n--- 특징: '{feature}' 분석 ---")

        feature_data = features_df[['species', feature]].copy().dropna()

        # 2. 각 그룹별 정규분포성 검정
        is_all_normal = True
        for species in species_list:
            data = feature_data[feature][feature_data['species'] == species]
            if len(data) >= 3:
                stat, p_value = shapiro(data)
                if p_value < 0.05:
                    is_all_normal = False
                    print(f"  - [경고] '{species}' 그룹의 데이터가 정규분포를 따르지 않습니다 (p={p_value:.4f}).")

        # 3. 분석 방법 결정 및 수행
        if is_all_normal:
            print("\n  -> 모든 그룹이 정규분포 가정을 만족하므로 ANOVA를 수행합니다.")
            grouped_data = [feature_data[feature][feature_data['species'] == s] for s in species_list]
            f_stat, p_val_anova = f_oneway(*grouped_data)
            print(f"     - ANOVA 결과: F={f_stat:.4f}, p-value={p_val_anova:.8f}")
            if p_val_anova < 0.05:
                print("     -> 결론: 종 간에 유의미한 차이가 있습니다.")
            else:
                print("     -> 결론: 유의미한 차이를 단정하기 어렵습니다.")

        else:
            print("\n  -> 일부 그룹이 정규분포를 따르지 않으므로, 두 가지 분석을 모두 수행합니다.")

            # 3-A: 데이터 변환 (로그 변환) 후 ANOVA
            print("\n    [방법 1] 로그 변환(Log Transformation) 후 ANOVA 수행")
            # 데이터에 0이나 음수가 있을 수 있으므로 np.log1p (log(1+x))를 사용
            transformed_feature = np.log1p(feature_data[feature] - feature_data[feature].min())

            grouped_transformed = [transformed_feature[feature_data['species'] == s] for s in species_list]

            f_stat_log, p_val_log_anova = f_oneway(*grouped_transformed)
            print(f"     - 변환 후 ANOVA 결과: F={f_stat_log:.4f}, p-value={p_val_log_anova:.8f}")
            if p_val_log_anova < 0.05:
                print("     -> 결론: 변환된 데이터에서 종 간에 유의미한 차이가 있습니다.")
            else:
                print("     -> 결론: 변환 후에도 유의미한 차이를 단정하기 어렵습니다.")

            # 3-B: 비모수적 검정 (크루스칼-월리스)
            print("\n    [방법 2] 비모수적 검정 (Kruskal-Wallis H-test) 수행")
            grouped_data = [feature_data[feature][feature_data['species'] == s] for s in species_list]
            h_stat, p_val_kruskal = kruskal(*grouped_data)
            print(f"     - Kruskal-Wallis 결과: H={h_stat:.4f}, p-value={p_val_kruskal:.8f}")
            if p_val_kruskal < 0.05:
                print("     -> 결론: 종 간에 유의미한 차이가 있습니다.")
            else:
                print("     -> 결론: 유의미한 차이를 단정하기 어렵습니다.")


def main():
    parser = argparse.ArgumentParser(description="데이터의 정규성을 검증하고, 비정규 데이터에 대해 변환 및 비모수 검정을 수행합니다.")
    parser.add_argument("input_csv", type=str, help="분석할 전체 특징 데이터가 담긴 CSV 파일 경로")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"오류: '{args.input_csv}' 파일을 찾을 수 없습니다.")
        return

    features_df = pd.read_csv(args.input_csv)
    advanced_statistical_analysis(features_df)


if __name__ == '__main__':
    main()
