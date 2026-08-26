import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import argparse
import os
from matplotlib.backends.backend_pdf import PdfPages

def normality_validation(features_df, output_pdf):
    """
    특징 데이터프레임을 받아 정규분포성 검정과 시각화를 수행합니다.
    """
    # 1. 데이터 준비
    feature_columns = features_df.select_dtypes(include=np.number).columns.tolist()
    species_list = features_df['species'].unique()
    
    print("="*60)
    print("각 형태 특징의 정규분포성 검증을 시작합니다.")
    print("="*60)
    
    normality_results = []

    # --- 2. 샤피로-윌크 검정 (Shapiro-Wilk Test) ---
    print("\n--- 샤피로-윌크 검정 결과 ---")
    print("P-value가 0.05보다 크면 정규분포를 따른다고 간주합니다.\n")
    
    for feature in feature_columns:
        for species in species_list:
            # 현재 종과 특징에 해당하는 데이터 추출
            data = features_df[features_df['species'] == species][feature].dropna()
            
            # 데이터가 3개 이상 있어야 검정 가능
            if len(data) >= 3:
                stat, p_value = shapiro(data)
                is_normal = p_value > 0.05
                normality_results.append({
                    'feature': feature,
                    'species': species,
                    'p_value': p_value,
                    'is_normal': is_normal
                })

    # 결과 요약표 출력
    results_df = pd.DataFrame(normality_results)
    # P-value를 소수점 4자리까지 표시
    results_df['p_value'] = results_df['p_value'].apply(lambda x: f"{x:.4f}")
    print(results_df.to_string())

    # 정규분포를 따르지 않는 데이터 요약
    non_normal_data = results_df[results_df['is_normal'] == False]
    if not non_normal_data.empty:
        print("\n\n--- [경고] 정규분포를 따르지 않는 데이터 목록 ---")
        print(non_normal_data.to_string(index=False))
    else:
        print("\n\n--- [정보] 모든 데이터가 정규분포 가정을 만족합니다. ---")

    # --- 3. 히스토그램 시각화 및 PDF 저장 ---
    print(f"\n데이터 분포 히스토그램을 '{output_pdf}' 파일로 저장합니다...")
    with PdfPages(output_pdf) as pdf:
        for feature in feature_columns:
            plt.figure(figsize=(12, 8))
            sns.histplot(data=features_df, x=feature, hue='species', kde=True, multiple="stack")
            plt.title(f'Distribution of "{feature}"', fontsize=16)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            pdf.savefig()  # 현재 페이지를 PDF에 저장
            plt.close() # 메모리에서 그래프 제거
    print("성공! 히스토그램 저장이 완료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="특징 데이터(CSV)의 정규분포성을 검증하고 시각화합니다.")
    parser.add_argument("input_csv", type=str, help="분석할 전체 특징 데이터가 담긴 CSV 파일 경로")
    parser.add_argument("output_pdf", type=str, help="결과 히스토그램을 저장할 PDF 파일 경로")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"오류: '{args.input_csv}' 파일을 찾을 수 없습니다.")
        return
        
    features_df = pd.read_csv(args.input_csv)
    normality_validation(features_df, args.output_pdf)

if __name__ == '__main__':
    main()
