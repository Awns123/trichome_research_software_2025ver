#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
advanced_trichome_analyzer.py

(Final Verified)
- 제안 1: 덴드로그램에 부트스트래핑(통계적 신뢰도) 기능 추가 (-n 옵션)
- 제안 2: PCA 분석에 Scree Plot 및 Loadings Plot 추가
- 제안 3: 전체 및 과별 산점도를 2가지 버전(Global Scale, Zoomed Scale)으로 출력
- 제안 4: 모든 결과물(좌표 CSV, 설명력 TXT 등) 자동 저장 및 축 고정 강화
- 제안 5: 범례(Legend)를 그래프에서 제거하고 별도 파일(pca_legend_family.png)로 저장
- 제안 6: 'Zoomed Scale' 그래프에서 이상치(Outlier) 제거 및 모든 과에 동일한 축 범위(Common Scale) 적용
- 제안 7: PCA 축 명칭을 'Principal Component' 풀네임으로 통일 (Loadings Plot 포함)
- 제안 8: 통합 PCA(Total PCA)의 축 범위를 과별 PCA(Zoomed)와 동일하게 맞춘 버전 추가
- 제안 9: 과별 PCA(Family PCA)에서도 범례를 제거하고 별도 파일(pca_legend_{Family}.png)로 저장
- 제안 10: 과별 확대(Zoomed) PCA에 대해서도 범례를 별도 파일(pca_legend_{Family}_zoomed.png)로 저장
- 제안 11: 모든 PCA 그래프 내부의 라벨링(범례) 완전 제거
- 제안 12: 확대(Zoomed) 그래프의 1:1 축 비율 설정 제거 (데이터 분포에 맞게 자동 조정)
"""
import os
import re
import glob
import argparse
import warnings
from typing import List, Tuple, Dict, Optional, FrozenSet

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.cluster.hierarchy import ClusterNode

# --- Font settings for Matplotlib ---
try:
    plt.rcParams['font.family'] = 'Arial'
except:
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        warnings.warn("Could not set a common font. Plot labels might not render correctly.")

# --- Family information for coloring PCA plot ---
FAMILY_MAPPING = {
    # 국화과 (Asteraceae)
    'A. alpina': 'Asteraceae',
    'D. oreastrum': 'Asteraceae',
    'T. erecta': 'Asteraceae',
    'G. pulchella': 'Asteraceae',
    'Calendula officinalis': 'Asteraceae',
    # 가지과 (Solanaceae)
    'S. nigrum': 'Solanaceae',
    'C. annuum': 'Solanaceae',
    'L. chinense': 'Solanaceae',
    'P. alkekengi': 'Solanaceae',
    'P. × hybrida': 'Solanaceae',
    'B. suaveolens': 'Solanaceae',
    # 쐐기풀과 (Urticaceae)
    'U. dioica': 'Urticaceae',
    'P. mollis': 'Urticaceae',
    'P. spruceana': 'Urticaceae',
    'P. involucrata': 'Urticaceae',
    'P. glauca': 'Urticaceae',
    'P. cadierei': 'Urticaceae'
}


# --- Data Processing Functions ---

def get_family(species_name: str, mapping: Dict[str, str] = FAMILY_MAPPING) -> str:
    """Returns the family of a species based on the mapping."""
    for key, family in mapping.items():
        if species_name.startswith(key):
            return family
    return 'Unknown'


def load_and_prepare_data(input_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads and merges CSVs from a directory, then preprocesses the data."""
    print(f"--- 1. Loading data from '{input_dir}' ---")
    all_data = []
    for filename in glob.glob(os.path.join(input_dir, '*.csv')):
        try:
            df = pd.read_csv(filename)
            species_name = os.path.basename(filename)
            species_name = os.path.splitext(species_name)[0]
            species_name = re.sub(r' \(.+\)| cv\..*| Mill\.$', '', species_name).strip()
            df['species'] = species_name
            all_data.append(df)
        except Exception as e:
            warnings.warn(f"Could not read {filename}. Skipping. Error: {e}")

    if not all_data:
        raise ValueError("No valid CSV files found in the input directory.")

    df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(df)} records from {len(all_data)} files.")

    features = df.select_dtypes(include=np.number).columns.tolist()
    features = [f for f in features if 'id' not in f.lower()]

    for col in features:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in '{col}' with mean ({mean_val:.2f}).")

    return df, features


def update_family_mapping_interactively(df: pd.DataFrame, mapping: Dict[str, str]):
    """Checks for unknown families and prompts the user to assign them."""
    unique_species = df['species'].unique()
    unknown_species = [s for s in unique_species if get_family(s, mapping) == 'Unknown']

    if not unknown_species:
        return

    print("\n--- [알림] 과(Family) 정보가 없는 종 발견 ---")
    print("아래 종들에 대한 과 정보를 입력해주세요.")

    known_families = {
        '1': 'Asteraceae', '2': 'Solanaceae', '3': 'Urticaceae'
    }

    for species in unknown_species:
        while True:
            prompt = (
                f"\n종 '{species}'은(는) 어떤 과에 속하나요?\n"
                "  1: 국화과 (Asteraceae)\n  2: 가지과 (Solanaceae)\n  3: 쐐기풀과 (Urticaceae)\n"
                "  또는, 직접 과 이름을 입력하세요: "
            )
            choice = input(prompt)

            if choice in known_families:
                family = known_families[choice]
                mapping[species] = family
                print(f"-> '{species}'을(를) '{family}'(으)로 지정했습니다.")
                break
            elif choice.strip():
                family = choice.strip()
                mapping[species] = family
                print(f"-> '{species}'을(를) '{family}'(으)로 지정했습니다.")
                break
            else:
                print("잘못된 입력입니다. 다시 시도해주세요.")


# --- Analysis & Visualization Functions ---

def save_family_legend(output_dir: str, family_colors: Dict[str, str], family_markers: Dict[str, str]):
    """
    (신규) 과(Family) 범례만 따로 이미지로 저장합니다.
    """
    print("  - Saving separate legend file...")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')

    # Create custom handles
    handles = []
    families = sorted(list(set(family_colors.keys()) & set(family_markers.keys())))

    # Filter only major families if they exist in colors
    major_families = ['Asteraceae', 'Solanaceae', 'Urticaceae']
    families = [f for f in major_families if f in family_colors] + [f for f in families if f not in major_families]

    for fam in families:
        if fam in family_colors and fam in family_markers:
            h = mlines.Line2D([], [], color=family_colors[fam], marker=family_markers[fam],
                              linestyle='None', markersize=10, label=fam)
            handles.append(h)

    if handles:
        ax.legend(handles=handles, title="Family", loc='center', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_legend_family.png"), bbox_inches='tight', dpi=300)
    plt.close()


def save_species_legend_from_ax(ax, output_path):
    """
    (신규) Plot의 Axes에서 범례 핸들과 라벨을 추출하여 별도 이미지로 저장합니다.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    # 범례 항목 수에 따라 이미지 높이 조절
    num_items = len(labels)
    fig_leg, ax_leg = plt.subplots(figsize=(5, 0.4 * num_items + 1))
    ax_leg.axis('off')

    ax_leg.legend(handles, labels, loc='center', frameon=True, ncol=1, title="Species")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig_leg)


def run_and_plot_pca(X_scaled: np.ndarray, df: pd.DataFrame, features: List[str],
                     output_dir: str, family_colors: Dict[str, str],
                     family_markers: Dict[str, str]):
    """
    (수정됨) PCA를 수행하고 범례 분리 및 축 범위가 통일된 그래프를 생성합니다.
    (Final) 모든 축 명칭을 'Principal Component X (Variance %)'로 통일하고 Zoomed Basic 버전을 추가합니다.
    """
    print("\n--- 4. Running PCA ---")

    pca = PCA(n_components=None)
    pca.fit(X_scaled)

    # 2D 산점도를 위해 PC1, PC2만 변환
    principal_components = pca.transform(X_scaled)[:, :2]

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['species'] = df['species']
    pca_df['family'] = df['species'].apply(get_family)

    explained_variance = pca.explained_variance_ratio_

    # [저장 기능] 좌표 및 설명력 저장
    csv_path = os.path.join(output_dir, "pca_results.csv")
    pca_df.to_csv(csv_path, index=False)

    var_path = os.path.join(output_dir, "pca_variance_info.txt")
    with open(var_path, "w") as f:
        f.write("PCA Explained Variance Ratio:\n")
        for i, var in enumerate(explained_variance):
            f.write(f"PC{i + 1}: {var * 100:.2f}%\n")
        f.write(f"\nCumulative Variance:\n")
        f.write(f"PC1+PC2: {np.sum(explained_variance[:2]) * 100:.2f}%\n")

    # --- [Step 1] Outlier Detection & Common Zoomed Scale Calculation ---
    Q1 = pca_df[['PC1', 'PC2']].quantile(0.25)
    Q3 = pca_df[['PC1', 'PC2']].quantile(0.75)
    IQR = Q3 - Q1

    outlier_factor = 2.5
    lower_bound = Q1 - outlier_factor * IQR
    upper_bound = Q3 + outlier_factor * IQR

    mask_robust = ((pca_df[['PC1', 'PC2']] >= lower_bound) & (pca_df[['PC1', 'PC2']] <= upper_bound)).all(axis=1)
    pca_df_robust = pca_df[mask_robust].copy()

    # Robust Scale (Zoomed 그래프용 공통 축 범위)
    padding = 0.5
    robust_xlim = (pca_df_robust['PC1'].min() - padding, pca_df_robust['PC1'].max() + padding)
    robust_ylim = (pca_df_robust['PC2'].min() - padding, pca_df_robust['PC2'].max() + padding)

    print(f"  > Robust Scale (Zoomed): X{robust_xlim}, Y{robust_ylim}")

    # --- [Step 2] Save Legend Separately ---
    save_family_legend(output_dir, family_colors, family_markers)

    # --- [Step 3] Plotting Auxiliary Graphs ---
    # 1. Scree Plot
    plt.figure(figsize=(10, 6))
    pc_components = np.arange(1, len(explained_variance) + 1)
    plt.bar(pc_components, explained_variance * 100, alpha=0.8, align='center')
    plt.plot(pc_components, np.cumsum(explained_variance * 100), 'r-o', label='Cumulative Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Scree Plot')
    plt.xticks(pc_components)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_plot_scree.png"))
    plt.close()

    # 2. Loadings Plot
    loadings = pca.components_
    pc1_loadings = loadings[0]
    pc2_loadings = loadings[1]

    plt.figure(figsize=(10, 8))
    plt.scatter(pc1_loadings, pc2_loadings, alpha=0.8)
    # [수정] 축 명칭 통일 (Loadings 제거 및 전체 통일)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
    plt.title('PCA Loadings Plot')
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)

    for i, feature in enumerate(features):
        plt.arrow(0, 0, pc1_loadings[i], pc2_loadings[i],
                  color='r', alpha=0.5, head_width=0.01)
        plt.text(pc1_loadings[i] * 1.05, pc2_loadings[i] * 1.05,
                 feature, ha='center', va='center', fontsize=9)
    plt.savefig(os.path.join(output_dir, "pca_plot_loadings.png"))
    plt.close()

    # Helper for Biplot vectors
    # Scale for Global
    max_score_global = np.max(np.abs(principal_components))
    max_loading = np.max(np.abs(loadings[:2]))
    scale_factor_global = max_score_global / max_loading * 0.8

    # Scale for Robust
    max_score_robust = np.max(np.abs(pca_df_robust[['PC1', 'PC2']].values))
    scale_factor_robust = max_score_robust / max_loading * 0.8

    def draw_biplot_vectors(ax, x_loads, y_loads, feats, scale):
        for i, feature in enumerate(feats):
            ax.arrow(0, 0, x_loads[i] * scale, y_loads[i] * scale,
                     color='red', alpha=0.4, head_width=scale * 0.03, zorder=10)
            ax.text(x_loads[i] * scale * 1.1, y_loads[i] * scale * 1.1,
                    feature, color='darkred', ha='center', va='center', fontsize=8, zorder=11)

    # --- [Step 4] Total PCA Plots ---

    # 4-1. Total PCA (Global Scale) - Includes Outliers, No Legend
    print("  - Generating Total Scatter Plots (Global Scale)...")
    fig_total, ax_total = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="PC1", y="PC2", hue="family", style="family",
        palette=family_colors, markers=family_markers,
        data=pca_df, s=80, alpha=0.8, ax=ax_total, legend=False
    )
    ax_total.set_title('Total PCA (Global Scale)')
    ax_total.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
    ax_total.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
    ax_total.grid(True)

    # Capture Global limits
    total_xlim = ax_total.get_xlim()
    total_ylim = ax_total.get_ylim()
    plt.savefig(os.path.join(output_dir, "pca_plot_total_global.png"), bbox_inches='tight')
    plt.show()

    # 4-2. Total PCA (Zoomed Scale) - Robust Limits, No Legend [NEW]
    # 과별 Zoomed 그래프와 범위를 맞춘 통합 그래프
    print("  - Generating Total Scatter Plots (Zoomed Scale)...")
    fig_robust, ax_robust = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="PC1", y="PC2", hue="family", style="family",
        palette=family_colors, markers=family_markers,
        data=pca_df, s=80, alpha=0.8, ax=ax_robust, legend=False
    )
    # Apply Robust Limits
    ax_robust.set_xlim(robust_xlim)
    ax_robust.set_ylim(robust_ylim)

    ax_robust.set_title('Total PCA (Zoomed Scale - Matches Family Plots)')
    ax_robust.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
    ax_robust.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
    ax_robust.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_plot_total_zoomed.png"), bbox_inches='tight')
    plt.show()

    # --- [Step 5] Family PCA Plots ---
    print("\n--- 5. Generating individual PCA plots for each family ---")
    families_to_plot = ['Asteraceae', 'Solanaceae', 'Urticaceae']

    for family in families_to_plot:
        if family not in pca_df['family'].unique():
            continue

        family_df_full = pca_df[pca_df['family'] == family]
        family_df_robust = pca_df_robust[pca_df_robust['family'] == family]

        # 5-1. Family (Global Scale) - Matches Total Global
        fig_fam1, ax_fam1 = plt.subplots(figsize=(10, 8))

        # [수정] 범례를 추출하기 위해 일단 legend='full'로 생성
        sns.scatterplot(
            x="PC1", y="PC2", hue="species", style="family",
            markers=family_markers, palette="viridis",
            data=family_df_full, s=100, alpha=0.9, ax=ax_fam1, legend='full'
        )

        # [수정] 범례 추출 및 별도 저장
        legend_path = os.path.join(output_dir, f"pca_legend_{family}.png")
        save_species_legend_from_ax(ax_fam1, legend_path)

        # [수정] 그래프 본체에서 범례 제거 (확실한 제거)
        if ax_fam1.get_legend():
            ax_fam1.get_legend().remove()

        ax_fam1.set_xlim(total_xlim)
        ax_fam1.set_ylim(total_ylim)

        ax_fam1.set_title(f'PCA - {family} (Global Scale)')
        ax_fam1.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
        ax_fam1.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
        ax_fam1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pca_plot_{family}_global.png"), bbox_inches='tight')
        plt.show()

        # 5-2. Family (Zoomed Scale Biplot) - Matches Total Zoomed (Robust Limits)
        fig_fam2, ax_fam2 = plt.subplots(figsize=(10, 8))
        if not family_df_robust.empty:
            # [수정] Zoomed 범례도 별도 저장을 위해 일단 legend='full'로 생성
            sns.scatterplot(
                x="PC1", y="PC2", hue="species", style="family",
                markers=family_markers, palette="viridis",
                data=family_df_robust, s=100, alpha=0.5, ax=ax_fam2, legend='full'
            )

            # [수정] Zoomed 범례 별도 저장 (파일명: pca_legend_{family}_zoomed.png)
            legend_zoomed_path = os.path.join(output_dir, f"pca_legend_{family}_zoomed.png")
            save_species_legend_from_ax(ax_fam2, legend_zoomed_path)

            # [수정] 그래프 본체에서 범례 제거 (확실한 제거)
            if ax_fam2.get_legend():
                ax_fam2.get_legend().remove()
        else:
            pass

        draw_biplot_vectors(ax_fam2, pc1_loadings, pc2_loadings, features, scale_factor_robust)

        # [중요] Robust 축 범위 적용 (Total Zoomed와 동일)
        ax_fam2.set_xlim(robust_xlim)
        ax_fam2.set_ylim(robust_ylim)

        ax_fam2.set_title(f'PCA - {family} (Zoomed Scale Biplot)')
        ax_fam2.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
        ax_fam2.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
        ax_fam2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pca_plot_{family}_zoomed_biplot.png"), bbox_inches='tight')
        plt.show()

        # 5-3. Family (Zoomed Scale Basic) - Matches Total Zoomed (Robust Limits, No Arrows) [NEW/RESTORED]
        fig_fam3, ax_fam3 = plt.subplots(figsize=(10, 8))
        if not family_df_robust.empty:
            sns.scatterplot(
                x="PC1", y="PC2", hue="species", style="family",
                markers=family_markers, palette="viridis",
                data=family_df_robust, s=100, alpha=0.9, ax=ax_fam3, legend=False
            )

        # [중요] Robust 축 범위 적용 (Total Zoomed와 동일)
        ax_fam3.set_xlim(robust_xlim)
        ax_fam3.set_ylim(robust_ylim)

        ax_fam3.set_title(f'PCA - {family} (Zoomed Scale Basic)')
        # [수정] 축 명칭 통일
        ax_fam3.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}%)')
        ax_fam3.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}%)')
        ax_fam3.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pca_plot_{family}_zoomed_basic.png"), bbox_inches='tight')
        plt.show()


def plot_dendrogram(Z: np.ndarray, labels: List[str], orient: str, font_size: int, out_path: str,
                    label_colors: Optional[Dict[str, str]] = None, title_suffix: str = ""):
    """(기존) 덴드로그램(.png)을 플롯하고 저장합니다."""
    if len(labels) < 2:
        warnings.warn(f"덴드로그램 생성 중단: 2종 이상의 데이터가 필요합니다.")
        return

    inch_per_leaf = 0.3
    max_fig_height = 40.0
    if orient in ['right', 'left']:
        fig_height = min(len(labels) * inch_per_leaf, max_fig_height)
        figsize = (15, fig_height)
    else:
        figsize = (20, 12)

    plt.figure(figsize=figsize)
    dendrogram(
        Z, orientation=orient, labels=labels,
        leaf_rotation=0 if orient in ['right', 'left'] else 90,
        leaf_font_size=font_size
    )
    ax = plt.gca()
    if orient in ['right', 'left']:
        tick_labels = ax.get_yticklabels()
        ax.set_xlabel('Distance (Ward)')
        ax.grid(axis='x')
    else:
        tick_labels = ax.get_xticklabels()
        ax.set_ylabel('Distance (Ward)')
        ax.grid(axis='y')
    if label_colors:
        for label in tick_labels:
            label_text = label.get_text()
            color = label_colors.get(label_text, 'black')
            label.set_color(color)
    plt.title(f'Hierarchical Clustering Dendrogram{title_suffix}')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Dendrogram saved to '{out_path}'")
    plt.show()


# --- [신규] 제안 1: 부트스트래핑 및 Newick 함수 ---

def get_all_clades_scipy(Z: np.ndarray, n_leaves: int) -> Dict[int, FrozenSet[int]]:
    """ (신규) Linkage 행렬(Z)로부터 모든 Clade와 잎 인덱스를 생성합니다. """
    clades = {i: frozenset([i]) for i in range(n_leaves)}
    for i, row in enumerate(Z):
        new_node_id = n_leaves + i
        node1_id = int(row[0])
        node2_id = int(row[1])
        new_clade = clades[node1_id].union(clades[node2_id])
        clades[new_node_id] = new_clade
    return clades


def run_ward_bootstrap(data_matrix: np.ndarray, n_bootstraps: int = 100
                       ) -> (Optional[np.ndarray], Dict[int, float]):
    """
    (신규) Ward + 부트스트래핑을 수행하고 Linkage 행렬과 신뢰도를 반환합니다.
    Ward는 'euclidean' 거리를 사용합니다.
    """
    n_species, n_features = data_matrix.shape

    if n_species < 2:
        warnings.warn(f"Ward 군집화 중단: 2종 이상의 데이터가 필요합니다.")
        return None, {}
    if n_species < 3:
        print(f"  (1/3) 원본 Ward 트리 생성 중 (2종)...")
        dist_orig = pdist(data_matrix, metric='euclidean')
        dist_orig = np.nan_to_num(dist_orig, nan=0.0, posinf=1.0, neginf=0.0)
        Z_orig = np.array([[0, 1, dist_orig[0], 2.0]])
        supports = {n_species: 100.0}
        print("  (2/3) 부트스트래핑 100회... (2종이므로 생략)")
        return Z_orig, supports

    print(f"  (1/3) 원본 Ward 트리 생성 중...")
    dist_orig = pdist(data_matrix, metric='euclidean')
    dist_orig = np.nan_to_num(dist_orig, nan=0.0, posinf=1.0, neginf=0.0)
    Z_orig = linkage(dist_orig, method='ward')

    original_clades_map = get_all_clades_scipy(Z_orig, n_species)
    original_internal_clades = {
        clade for clade in original_clades_map.values() if len(clade) > 1
    }
    clade_counts = {clade: 0 for clade in original_internal_clades}

    print(f"  (2/3) 부트스트래핑 {n_bootstraps}회 반복 수행 중...")
    valid_bootstraps = 0
    for i in range(n_bootstraps):
        if (i + 1) % (n_bootstraps / 10 if n_bootstraps >= 10 else 1) == 0:
            print(f"    ... {i + 1} / {n_bootstraps} 완료")

        indices = np.random.choice(n_features, n_features, replace=True)
        boot_matrix = data_matrix[:, indices]

        dist_boot = pdist(boot_matrix, metric='euclidean')
        dist_boot = np.nan_to_num(dist_boot, nan=0.0, posinf=1.0, neginf=0.0)

        try:
            if np.all(dist_boot == 0):
                raise ValueError("모든 거리가 0입니다.")
            Z_boot = linkage(dist_boot, method='ward')
            valid_bootstraps += 1
        except Exception as e:
            warnings.warn(f"부트스트랩 {i}회차 linkage 실패 (skip): {e}")
            continue

        boot_clades_map = get_all_clades_scipy(Z_boot, n_species)
        boot_internal_clades = {
            clade for clade in boot_clades_map.values() if len(clade) > 1
        }

        for clade in original_internal_clades:
            if clade in boot_internal_clades:
                clade_counts[clade] += 1

    print(f"  (3/3) 신뢰도 값 매핑 중 ({valid_bootstraps}회 유효)...")
    bootstrap_supports = {}
    for node_id, clade in original_clades_map.items():
        if node_id >= n_species:
            count = clade_counts.get(clade, 0)
            if valid_bootstraps > 0:
                bootstrap_supports[node_id] = (count / valid_bootstraps) * 100.0
            else:
                bootstrap_supports[node_id] = 0.0

    return Z_orig, bootstrap_supports


def get_newick_recursive(node: ClusterNode, labels: List[str], parent_dist: float,
                         bootstrap_supports: Dict[int, float]) -> str:
    """
    (수정됨) Newick 문자열 변환 + 부트스트랩 신뢰도 값 추가
    """
    if node.is_leaf():
        name = labels[node.id]
        clean_name = re.sub(r'[\s:(),\[\]×]+', '_', name)
        branch_length = max(0.0, parent_dist)
        return f"{clean_name}:{branch_length:.5f}"
    else:
        branch_length = max(0.0, parent_dist - node.dist)

        left_subtree = get_newick_recursive(
            node.get_left(), labels, node.dist, bootstrap_supports
        )
        right_subtree = get_newick_recursive(
            node.get_right(), labels, node.dist, bootstrap_supports
        )

        # 신뢰도 값 추가
        support_val = bootstrap_supports.get(node.id, 0.0)
        return f"({left_subtree},{right_subtree}){support_val:.0f}:{branch_length:.5f}"


def save_newick_tree(Z: np.ndarray, labels: List[str], out_path: str,
                     bootstrap_supports: Dict[int, float]):
    """
    (수정됨) Linkage 행렬(Z)을 부트스트랩 신뢰도가 포함된 Newick 파일로 저장합니다.
    """
    print(f"Converting to Newick tree format...")
    try:
        if len(labels) == 2:
            support_val = bootstrap_supports.get(len(labels), 100.0)
            dist = Z[0, 2]
            name1 = re.sub(r'[\s:(),\[\]×]+', '_', labels[0])
            name2 = re.sub(r'[\s:(),\[\]×]+', '_', labels[1])
            newick_string = f"({name1}:{dist:.5f},{name2}:{dist:.5f}){support_val:.0f};"
        else:
            root_node = to_tree(Z)
            left_subtree = get_newick_recursive(
                root_node.get_left(), labels, root_node.dist, bootstrap_supports
            )
            right_subtree = get_newick_recursive(
                root_node.get_right(), labels, root_node.dist, bootstrap_supports
            )
            newick_string = f"({left_subtree},{right_subtree});"

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(newick_string)
        print(f"Newick tree saved to '{out_path}'")

    except Exception as e:
        warnings.warn(f"Could not generate Newick tree '{out_path}'. Error: {e}")


# --- [END NEW FUNCTION BLOCK] ---


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of trichome morphometrics.")
    parser.add_argument("input_dir", help="Directory containing the species CSV files.")
    parser.add_argument("output_dir", help="Directory to save the analysis results.")
    parser.add_argument("--mode", choices=['individual', 'species'], default='species',
                        help="Clustering mode: 'individual' (all samples) or 'species' (species averages).")
    parser.add_argument("--orient", choices=['top', 'right', 'bottom', 'left'], default='right',
                        help="Orientation of the dendrogram.")
    parser.add_argument("-n", "--n_bootstraps", type=int, default=100,
                        help="Number of bootstrap replicates (default: 100)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, features = load_and_prepare_data(args.input_dir)
    update_family_mapping_interactively(df, FAMILY_MAPPING)

    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    family_colors = {
        'Asteraceae': 'red', 'Solanaceae': 'blue', 'Urticaceae': 'green', 'Unknown': 'black'
    }
    for family in df['species'].apply(get_family).unique():
        if family not in family_colors:
            family_colors[family] = 'purple'
    family_markers = {
        'Asteraceae': 'o', 'Solanaceae': 's', 'Urticaceae': '^', 'Unknown': 'X'
    }
    for family in df['species'].apply(get_family).unique():
        if family not in family_markers:
            family_markers[family] = 'P'

    # (수정) PCA 및 2가지 버전의 산점도 출력 함수 실행
    run_and_plot_pca(X_scaled, df, features, args.output_dir, family_colors, family_markers)

    if args.mode == 'species':
        print(f"\n--- 6. Running Hierarchical Clustering in '{args.mode}' mode (All Species) ---")
        species_mean = df.groupby('species')[features].mean(numeric_only=True)
        X_species_scaled = StandardScaler().fit_transform(species_mean)

        # (수정) 부트스트래핑 수행
        Z, supports = run_ward_bootstrap(
            X_species_scaled, n_bootstraps=args.n_bootstraps
        )

        if Z is None:
            print("Clustering for all species failed.")
        else:
            labels = species_mean.index.tolist()
            label_colors = {label: family_colors.get(get_family(label), 'black') for label in labels}

            out_dendro_png = os.path.join(args.output_dir, "dendrogram_species_ALL.png")
            plot_dendrogram(Z, labels, args.orient, 10, out_dendro_png,
                            label_colors=label_colors, title_suffix=" (All Species)")

            # (수정) 신뢰도 값이 포함된 Newick 파일 저장
            out_dendro_nwk = os.path.join(args.output_dir, "dendrogram_species_bootstrap_ALL.nwk")
            save_newick_tree(Z, labels, out_dendro_nwk, supports)

        # --- (수정) 과별 덴드로그램 생성 로직 ---
        print(f"\n--- 7. Running Hierarchical Clustering for each major family ---")
        families_to_plot = ['Asteraceae', 'Solanaceae', 'Urticaceae']

        for family in families_to_plot:
            print(f"\nProcessing dendrogram for: {family}")

            family_species_labels = [
                s for s in species_mean.index if get_family(s, FAMILY_MAPPING) == family
            ]

            if len(family_species_labels) < 2:
                print(
                    f"Skipping dendrogram for {family}: requires at least 2 species, found {len(family_species_labels)}.")
                continue

            family_data = species_mean.loc[family_species_labels]

            X_family_scaled = StandardScaler().fit_transform(family_data)

            # (수정) 과별 부트스트래핑 수행
            Z_family, supports_family = run_ward_bootstrap(
                X_family_scaled, n_bootstraps=args.n_bootstraps
            )

            if Z_family is None:
                print(f"Clustering for {family} failed.")
                continue

            family_color = family_colors.get(family, 'black')
            label_colors_family = {label: family_color for label in family_species_labels}

            out_dendro_family_png = os.path.join(
                args.output_dir, f"dendrogram_species_{family}.png"
            )
            plot_dendrogram(
                Z_family,
                family_species_labels,
                args.orient,
                10,
                out_dendro_family_png,
                label_colors=label_colors_family,
                title_suffix=f" ({family} only)"
            )

            # (수정) 신뢰도 값이 포함된 과별 Newick 파일 저장
            out_dendro_family_nwk = os.path.join(
                args.output_dir, f"dendrogram_species_bootstrap_{family}.nwk"
            )
            save_newick_tree(Z_family, family_species_labels, out_dendro_family_nwk, supports_family)
        # --- [END MODIFIED BLOCK] ---

    print("\n--- Analysis complete ---")


if __name__ == '__main__':
    main()