import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def import_sascorer():
    """
    导入 RDKit Contrib 中的 SA_Score 工具。
    不同 RDKit 安装方式下路径可能不同，因此这里做了兼容处理。
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except ImportError:
        try:
            from rdkit import RDConfig
            sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
            sys.path.append(sa_score_path)
            import sascorer
            return sascorer
        except Exception as e:
            raise ImportError(
                "无法导入 RDKit 的 sascorer。\n"
                "建议使用 conda 安装 RDKit：\n"
                "conda install -c conda-forge rdkit pandas\n\n"
                f"原始错误：{e}"
            )


sascorer = import_sascorer()


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def count_stereocenters(mol):
    stereo_centers = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False
    )
    return len(stereo_centers)


def count_large_rings(mol, min_ring_size=8):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    return sum(1 for ring in atom_rings if len(ring) >= min_ring_size)


def get_risk_functional_groups(mol):
    """
    一些可能带来合成、稳定性或后处理风险的官能团。
    这些不是绝对禁用规则，而是用于提醒。
    """
    smarts_dict = {
        "peroxide": "[OX2][OX2]",
        "azide": "[N-]=[N+]=[N]",
        "diazo": "[C]=[N+]=[N-]",
        "acid_chloride": "[CX3](=O)[Cl]",
        "anhydride": "[CX3](=O)O[CX3](=O)",
        "isocyanate": "N=C=O",
        "isothiocyanate": "N=C=S",
        "epoxide": "C1OC1",
        "oxime": "[CX3]=[NX2][OX2H0,OX2H1]",
        "aldehyde": "[CX3H1](=O)[#6]",
        "free_thiol": "[SX2H]",
        "o_f_bond": "[O][F]",
        "n_f_bond": "[N][F]",
    }

    found = []

    for name, smarts in smarts_dict.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            found.append(name)

    return found


def classify_sa_score(sa_score):
    """
    SA Score 越低越容易合成。
    这里的分级是经验分级，适合初筛。
    """
    if sa_score <= 3:
        return "easy"
    elif sa_score <= 5:
        return "moderate"
    elif sa_score <= 7:
        return "hard"
    else:
        return "very_hard"


def calculate_synthetic_penalty(row):
    """
    构造一个额外的合成复杂度惩罚。
    它不是文献标准分数，而是便于分子库排序的工程化指标。
    """
    penalty = 0

    if row["num_stereocenters"] >= 2:
        penalty += row["num_stereocenters"] - 1

    if row["num_spiro_atoms"] > 0:
        penalty += 1.5 * row["num_spiro_atoms"]

    if row["num_bridgehead_atoms"] > 0:
        penalty += 1.5 * row["num_bridgehead_atoms"]

    if row["num_large_rings"] > 0:
        penalty += 1.0 * row["num_large_rings"]

    if row["num_rings"] >= 4:
        penalty += 1.0

    if row["rotatable_bonds"] >= 10:
        penalty += 0.5

    if row["risk_group_count"] > 0:
        penalty += 1.5 * row["risk_group_count"]

    if row["mol_weight"] > 600:
        penalty += 1.0

    return penalty


def evaluate_one_molecule(smiles):
    canonical = canonicalize_smiles(smiles)

    if canonical is None:
        return None

    mol = Chem.MolFromSmiles(canonical)

    sa_score = float(sascorer.calculateScore(mol))

    risk_groups = get_risk_functional_groups(mol)

    result = {
        "canonical_smiles": canonical,
        "sa_score": sa_score,
        "sa_class": classify_sa_score(sa_score),
        "mol_weight": Descriptors.MolWt(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "num_large_rings": count_large_rings(mol, min_ring_size=8),
        "num_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "num_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "num_stereocenters": count_stereocenters(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "h_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "risk_groups": ";".join(risk_groups),
        "risk_group_count": len(risk_groups),
    }

    return result


def evaluate_csv(
    input_csv,
    smiles_col="SMILES",
    output_dir="synthesizability_results",
    deduplicate=True
):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if smiles_col not in df.columns:
        raise ValueError(
            f"找不到 SMILES 列：{smiles_col}\n"
            f"当前 CSV 中的列有：{list(df.columns)}"
        )

    df["canonical_smiles"] = df[smiles_col].apply(canonicalize_smiles)

    invalid_df = df[df["canonical_smiles"].isna()].copy()
    valid_df = df.dropna(subset=["canonical_smiles"]).copy()

    if deduplicate:
        valid_df = valid_df.drop_duplicates(subset=["canonical_smiles"]).copy()

    records = []

    for _, row in valid_df.iterrows():
        smiles = row["canonical_smiles"]
        result = evaluate_one_molecule(smiles)

        if result is None:
            continue

        original_data = row.to_dict()

        for key, value in result.items():
            original_data[key] = value

        records.append(original_data)

    result_df = pd.DataFrame(records)

    if len(result_df) > 0:
        result_df["synthetic_penalty"] = result_df.apply(
            calculate_synthetic_penalty,
            axis=1
        )

        result_df["synthesizability_score"] = (
            result_df["sa_score"] + result_df["synthetic_penalty"]
        )

        result_df = result_df.sort_values(
            by=["synthesizability_score", "sa_score", "risk_group_count"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

        result_df.insert(0, "synth_rank", range(1, len(result_df) + 1))

    output_csv = output_dir / "synthesizability_evaluation.csv"
    invalid_csv = output_dir / "invalid_smiles.csv"
    summary_csv = output_dir / "synthesizability_summary.csv"

    result_df.to_csv(output_csv, index=False)
    invalid_df.to_csv(invalid_csv, index=False)

    if len(result_df) > 0:
        summary = result_df["sa_class"].value_counts().reset_index()
        summary.columns = ["sa_class", "count"]
        summary["ratio"] = summary["count"] / len(result_df)
        summary.to_csv(summary_csv, index=False)

    print("可合成性评估完成")
    print("=" * 50)
    print(f"输入文件：{input_csv}")
    print(f"原始分子数：{len(df)}")
    print(f"有效分子数：{len(valid_df)}")
    print(f"无效 SMILES 数：{len(invalid_df)}")

    if len(result_df) > 0:
        print(f"SA Score 平均值：{result_df['sa_score'].mean():.3f}")
        print(f"SA Score 中位数：{result_df['sa_score'].median():.3f}")
        print(f"最容易合成 SA Score：{result_df['sa_score'].min():.3f}")
        print(f"最难合成 SA Score：{result_df['sa_score'].max():.3f}")
        print()
        print("SA 分级统计：")
        print(result_df["sa_class"].value_counts())

    print("=" * 50)
    print(f"详细结果：{output_csv}")
    print(f"无效 SMILES：{invalid_csv}")
    print(f"分级统计：{summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="评估 SMILES 分子库的可合成性"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="输入 CSV 文件路径"
    )

    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="SMILES 所在列名，默认是 SMILES"
    )

    parser.add_argument(
        "--output_dir",
        default="synthesizability_results",
        help="输出文件夹"
    )

    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="不对 canonical SMILES 去重"
    )

    args = parser.parse_args()

    evaluate_csv(
        input_csv=args.input,
        smiles_col=args.smiles_col,
        output_dir=args.output_dir,
        deduplicate=not args.no_deduplicate
    )


if __name__ == "__main__":
    main()