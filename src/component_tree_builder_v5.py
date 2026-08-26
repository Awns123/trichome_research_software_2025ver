#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
component_tree_builder_final.py

(SMC, Jaccard + UPGMA 전용)
Biopython/NJ의 반복되는 오류로 인해 해당 기능을 모두 제거하고,
Scipy만을 사용하여 UPGMA + 부트스트래핑을 수행하는 안정화 버전입니다.

기능:
1. SMC + UPGMA (전체 종, 과별)
2. Jaccard + UPGMA (전체 종, 과별)
3. 'Unknown' 종에 대한 대화형 과(Family) 지정 프롬프트
"""

import os
import re
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, ClusterNode
from typing import List, Set, Dict, FrozenSet

# --- Matplotlib 글꼴 설정 ---
try:
    plt.rcParams['font.family'] = 'Arial'
except:
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        warnings.warn("Arial 또는 DejaVu Sans 글꼴을 찾을 수 없습니다. 라벨이 깨질 수 있습니다.")

# --- (신규) Family 정보 추가 ---
FAMILY_MAPPING = {
    # 국화과 (Asteraceae)
    'A. alpina': 'Asteraceae',
    'D. oreastrum': 'Asteraceae',
    'T. erecta': 'Asteraceae',
    'G. pulchella': 'Asteraceae',
    'Calendula officinalis': 'Asteraceae',
    'C. cyanus': 'Asteraceae',
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
    'P. cadierei': 'Urticaceae',
    'P. peperomioides': 'Urticaceae',
    # CSV 파일 이름과 매핑 키를 일치시켜야 함
    'A. dioica (L.) Gaertn': 'Asteraceae',
    'A. alpina subsp. rhodoptarmica (Nakai) Kitam.': 'Asteraceae',
    'C. annuum var. grossum': 'Solanaceae',
    'L. chinense Mill.': 'Solanaceae',
    'P. alkekengi var. franchetii': 'Solanaceae',
    'P. × hybrida ‘Dreams Red’': 'Solanaceae',
    'P. mollis cv. \'Moon Valley\'': 'Urticaceae',
    'P. cadierei Gagnep. & Guillaumin': 'Urticaceae',
    'Urtica dioica L.': 'Urticaceae'
}


def get_family(species_name: str, mapping: Dict[str, str] = FAMILY_MAPPING) -> str:
    """ 종 이름을 기반으로 과(Family) 이름을 반환합니다. """
    if species_name in mapping:
        return mapping[species_name]
    for key, family in mapping.items():
        if species_name.startswith(key):
            return family
    return 'Unknown'


def update_family_mapping_interactively(df_transposed: pd.DataFrame, mapping: Dict[str, str]):
    """ 'Unknown' 종에 대해 사용자에게 입력을 받아 매핑을 업데이트합니다. """
    unique_species = df_transposed.index.tolist()
    unknown_species = [s for s in unique_species if get_family(s, mapping) == 'Unknown']

    if not unknown_species:
        return

    print(f"\n--- [알림] {len(unknown_species)}개의 과(Family) 정보가 없는 종 발견 ---")
    print("과별 분석을 위해 아래 종들에 대한 과 정보를 입력해주세요.")

    known_families = {
        '1': 'Asteraceae', '2': 'Solanaceae', '3': 'Urticaceae'
    }

    for species in unknown_species:
        while True:
            prompt = (
                f"\n종 '{species}'은(는) 어떤 과에 속하나요?\n"
                "  1: 국화과 (Asteraceae)\n"
                "  2: 가지과 (Solanaceae)\n"
                "  3: 쐐기풀과 (Urticaceae)\n"
                "  4: 이 종을 '과별 분석'에서 제외 (Unknown으로 유지)\n"
                "  또는, 직접 과 이름을 입력하세요: "
            )
            choice = input(prompt).strip()

            if choice in known_families:
                family = known_families[choice]
                mapping[species] = family
                print(f"-> '{species}'을(를) '{family}'(으)로 지정했습니다.")
                break
            elif choice == '4':
                print(f"-> '{species}'을(를) 'Unknown'으로 유지합니다.")
                break
            elif choice:
                family = choice
                mapping[species] = family
                print(f"-> '{species}'을(를) '{family}'(으)로 지정했습니다.")
                break
            else:
                print("잘못된 입력입니다. 다시 시도해주세요.")


# --- Family 정보 끝 ---


# ==============================================================================
# 섹션 1: UPGMA (scipy) 관련 헬퍼 함수
# ==============================================================================

def get_all_clades_scipy(Z: np.ndarray, n_leaves: int) -> Dict[int, FrozenSet[int]]:
    clades = {i: frozenset([i]) for i in range(n_leaves)}
    for i, row in enumerate(Z):
        new_node_id = n_leaves + i
        node1_id = int(row[0])
        node2_id = int(row[1])
        new_clade = clades[node1_id].union(clades[node2_id])
        clades[new_node_id] = new_clade
    return clades


def run_upgma_bootstrap(data_matrix: np.ndarray, metric: str, n_bootstraps: int = 100
                        ) -> (np.ndarray, Dict[int, float]):
    n_species, n_features = data_matrix.shape

    if n_species < 2:
        warnings.warn(f"UPGMA 중단: 2종 이상의 데이터가 필요합니다 (현재 {n_species}종)")
        return None, {}
    if n_species < 3:
        print(f"  (1/3) 원본 {metric} + UPGMA 트리 생성 중 (2종)...")
        dist_orig = pdist(data_matrix, metric=metric)
        dist_orig = np.nan_to_num(dist_orig, nan=0.0, posinf=1.0, neginf=0.0)
        Z_orig = np.array([[0, 1, dist_orig[0], 2.0]])
        supports = {n_species: 100.0}
        print("  (2/3) 부트스트래핑 100회... (2종이므로 생략)")
        print("  (3/3) 신뢰도 값 매핑 중...")
        return Z_orig, supports

    print(f"  (1/3) 원본 {metric} + UPGMA 트리 생성 중...")
    dist_orig = pdist(data_matrix, metric=metric)
    dist_orig = np.nan_to_num(dist_orig, nan=0.0, posinf=1.0, neginf=0.0)
    Z_orig = linkage(dist_orig, method='average')

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

        dist_boot = pdist(boot_matrix, metric=metric)
        dist_boot = np.nan_to_num(dist_boot, nan=0.0, posinf=1.0, neginf=0.0)

        try:
            if np.all(dist_boot == 0):
                raise ValueError("모든 거리가 0입니다.")
            Z_boot = linkage(dist_boot, method='average')
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


def get_newick_recursive_scipy(node: ClusterNode, labels: List[str], parent_dist: float,
                               bootstrap_supports: Dict[int, float]) -> str:
    if node.is_leaf():
        name = labels[node.id]
        clean_name = re.sub(r'[\s:(),\[\]×\.]+', '_', name)
        branch_length = max(0.0, parent_dist)
        return f"{clean_name}:{branch_length:.6f}"
    else:
        branch_length = max(0.0, parent_dist - node.dist)
        left_subtree = get_newick_recursive_scipy(
            node.get_left(), labels, node.dist, bootstrap_supports
        )
        right_subtree = get_newick_recursive_scipy(
            node.get_right(), labels, node.dist, bootstrap_supports
        )
        support_val = bootstrap_supports.get(node.id, 0.0)
        return f"({left_subtree},{right_subtree}){support_val:.0f}:{branch_length:.6f}"


def save_newick_tree_scipy(Z: np.ndarray, labels: List[str], out_path: str,
                           bootstrap_supports: Dict[int, float]):
    print(f"Newick 트리 형식으로 변환 중...")
    try:
        if len(labels) == 2:
            support_val = bootstrap_supports.get(len(labels), 100.0)
            dist = Z[0, 2] / 2.0
            name1 = re.sub(r'[\s:(),\[\]×\.]+', '_', labels[0])
            name2 = re.sub(r'[\s:(),\[\]×\.]+', '_', labels[1])
            newick_string = f"({name1}:{dist:.6f},{name2}:{dist:.6f}){support_val:.0f};"
        else:
            root_node = to_tree(Z)
            left_subtree = get_newick_recursive_scipy(
                root_node.get_left(), labels, root_node.dist, bootstrap_supports
            )
            right_subtree = get_newick_recursive_scipy(
                root_node.get_right(), labels, root_node.dist, bootstrap_supports
            )
            newick_string = f"({left_subtree},{right_subtree});"

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(newick_string)
        print(f"Newick 트리를 '{out_path}'에 저장했습니다.")

    except Exception as e:
        warnings.warn(f"'{out_path}' Newick 트리 생성에 실패했습니다. 오류: {e}")


def plot_dendrogram_scipy(Z: np.ndarray, labels: List[str], out_path_png: str, title: str):
    if len(labels) < 2:
        warnings.warn(f"덴드로그램 생성 중단: 2종 이상의 데이터가 필요합니다.")
        return

    plt.figure(figsize=(14, len(labels) * 0.4 + 1))
    dendrogram(
        Z, labels=labels, orientation='right', leaf_font_size=10
    )
    plt.title(title)
    plt.xlabel('Distance')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(out_path_png)
    print(f"덴드로그램 이미지를 '{out_path_png}'에 저장했습니다.")
    plt.show()


# ==============================================================================
# 메인 실행 함수
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="성분 데이터(.csv)로부터 UPGMA 계통도를 생성합니다 (SMC, Jaccard 지원)."
    )
    parser.add_argument("input_file",
                        help="입력 CSV 파일 경로. (예: 전형질 분석_성분.csv)")
    parser.add_argument("output_dir",
                        help="결과 파일(png, nwk)을 저장할 디렉터리.")
    parser.add_argument("-n", "--n_bootstraps", type=int, default=100,
                        help="부트스트래핑 반복 횟수 (기본값: 100)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. 데이터 불러오기 및 전처리 ---
    print(f"--- 1. '{args.input_file}' 파일 불러오는 중 ---")
    try:
        df = pd.read_csv(args.input_file, index_col=0, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input_file, index_col=0, encoding='utf-8')
    except FileNotFoundError:
        print(f"오류: '{args.input_file}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return

    df_transposed = df.T
    df_transposed.index = df_transposed.index.str.strip()

    all_species_labels = df_transposed.index.tolist()
    all_data_matrix = df_transposed.values.astype(int)

    # --- (신규) 대화형 Family 정보 업데이트 ---
    update_family_mapping_interactively(df_transposed, FAMILY_MAPPING)

    species_to_family = {
        s: get_family(s, FAMILY_MAPPING) for s in all_species_labels
    }
    # ---

    print(f"\n데이터 불러오기 완료: {len(all_species_labels)}개 종, {len(df.index)}개 성분")
    print(f"부트스트래핑 반복 횟수: {args.n_bootstraps}")

    # --- 2. (전체 종) UPGMA 분석 (SMC, Jaccard) ---
    print(f"\n--- 2. (전체 종) UPGMA 분석 수행 (N={args.n_bootstraps}) ---")

    for metric in ['matching', 'jaccard']:
        metric_name = "SMC" if metric == 'matching' else "Jaccard"
        print(f"\n--- 2-{metric.upper()}: {metric_name} + UPGMA (전체) ---")

        linkage_all, supports_all = run_upgma_bootstrap(
            all_data_matrix, metric=metric, n_bootstraps=args.n_bootstraps
        )

        if linkage_all is not None:
            out_png = os.path.join(args.output_dir, f"dendrogram_upgma_{metric}_ALL.png")
            plot_dendrogram_scipy(linkage_all, all_species_labels, out_png,
                                  f'UPGMA ({metric_name}) - ALL SPECIES')

            out_nwk = os.path.join(args.output_dir, f"tree_upgma_{metric}_bootstrap_ALL.nwk")
            save_newick_tree_scipy(linkage_all, all_species_labels, out_nwk, supports_all)

    # --- (신규) 섹션 3: 과(Family)별 분석 ---
    print(f"\n--- 3. (과별) UPGMA 분석 수행 (N={args.n_bootstraps}) ---")

    families_in_data = set(species_to_family.values())
    families_to_analyze = [f for f in families_in_data if f != 'Unknown']

    for family in families_to_analyze:
        print(f"\n==========================================")
        print(f"  과(Family) 분석 시작: {family}")
        print(f"==========================================")

        family_species_labels = [
            s for s in all_species_labels if species_to_family[s] == family
        ]
        family_species_indices = [
            i for i, s in enumerate(all_species_labels) if species_to_family[s] == family
        ]

        family_data_matrix = all_data_matrix[family_species_indices, :]

        print(f"  {family} 소속 {len(family_species_labels)}개 종 발견.")

        # --- 과별 UPGMA (SMC, Jaccard) ---
        for metric in ['matching', 'jaccard']:
            metric_name = "SMC" if metric == 'matching' else "Jaccard"
            print(f"\n--- 3-{metric.upper()}: {metric_name} + UPGMA ({family}) ---")

            link_fam, sup_fam = run_upgma_bootstrap(
                family_data_matrix, metric=metric, n_bootstraps=args.n_bootstraps
            )

            if link_fam is not None:
                out_png_fam = os.path.join(args.output_dir, f"dendrogram_upgma_{metric}_{family}.png")
                plot_dendrogram_scipy(link_fam, family_species_labels, out_png_fam,
                                      f'UPGMA ({metric_name}) - {family} only')

                out_nwk_fam = os.path.join(args.output_dir, f"tree_upgma_{metric}_bootstrap_{family}.nwk")
                save_newick_tree_scipy(link_fam, family_species_labels, out_nwk_fam, sup_fam)

    print("\n--- 모든 분석 완료 ---")


if __name__ == '__main__':
    main()