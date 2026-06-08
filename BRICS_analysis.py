import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Draw


def canonicalize_smiles(smiles):
    """
    标准化 SMILES。
    无效 SMILES 返回 None。
    """
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def normalize_dummy_atoms(fragment_smiles, keep_dummy_labels=False):
    """
    BRICS 分解后会出现类似 [1*]、[3*] 的连接位点。
    
    如果 keep_dummy_labels=False：
        [1*]、[3*]、[16*] 都统一成 [*]
        更适合统计“片段本身”出现频率。

    如果 keep_dummy_labels=True：
        保留 BRICS 连接位点类型。
        更适合分析“可连接方式”。
    """
    if keep_dummy_labels:
        return fragment_smiles

    return re.sub(r"\[\d+\*\]", "[*]", fragment_smiles)


def extract_brics_fragments(mol, keep_dummy_labels=False):
    """
    使用 BRICS 规则切断分子，得到分子片段。
    """
    try:
        broken_mol = BRICS.BreakBRICSBonds(mol)
        frag_mols = Chem.GetMolFrags(
            broken_mol,
            asMols=True,
            sanitizeFrags=True
        )

        fragments = []

        for frag_mol in frag_mols:
            frag_smi = Chem.MolToSmiles(frag_mol, canonical=True)
            frag_smi = normalize_dummy_atoms(
                frag_smi,
                keep_dummy_labels=keep_dummy_labels
            )

            frag_mol_check = Chem.MolFromSmiles(frag_smi)
            if frag_mol_check is not None:
                frag_smi = Chem.MolToSmiles(frag_mol_check, canonical=True)
                fragments.append(frag_smi)

        return fragments

    except Exception:
        return []


def count_fragments(
    input_csv,
    smiles_col="SMILES",
    output_dir="fragment_frequency_results",
    min_mol_freq=2,
    top_n=50,
    keep_dummy_labels=False,
    deduplicate_molecules=True
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

    if deduplicate_molecules:
        valid_df = valid_df.drop_duplicates(subset=["canonical_smiles"]).copy()

    total_counter = Counter()
    molecule_counter = Counter()

    example_molecule = {}
    example_row_index = {}

    fragment_to_molecules = defaultdict(set)

    for row_idx, row in valid_df.iterrows():
        smiles = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fragments = extract_brics_fragments(
            mol,
            keep_dummy_labels=keep_dummy_labels
        )

        if len(fragments) == 0:
            continue

        total_counter.update(fragments)

        unique_fragments_in_molecule = set(fragments)
        molecule_counter.update(unique_fragments_in_molecule)

        for frag in unique_fragments_in_molecule:
            fragment_to_molecules[frag].add(smiles)

            if frag not in example_molecule:
                example_molecule[frag] = smiles
                example_row_index[frag] = row_idx

    records = []

    for frag_smi, mol_freq in molecule_counter.items():
        total_occurrence = total_counter[frag_smi]

        if mol_freq < min_mol_freq:
            continue

        mol = Chem.MolFromSmiles(frag_smi)

        if mol is not None:
            heavy_atom_count = mol.GetNumHeavyAtoms()
            atom_count = mol.GetNumAtoms()
        else:
            heavy_atom_count = None
            atom_count = None

        records.append({
            "fragment_smiles": frag_smi,
            "molecule_frequency": mol_freq,
            "total_occurrence": total_occurrence,
            "heavy_atom_count": heavy_atom_count,
            "atom_count": atom_count,
            "example_molecule_smiles": example_molecule.get(frag_smi),
            "example_row_index": example_row_index.get(frag_smi),
        })

    result_df = pd.DataFrame(records)

    if len(result_df) > 0:
        result_df = result_df.sort_values(
            by=["molecule_frequency", "total_occurrence", "heavy_atom_count"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        result_df.insert(0, "rank", range(1, len(result_df) + 1))

    output_csv = output_dir / "fragment_frequency.csv"
    result_df.to_csv(output_csv, index=False)

    invalid_output_csv = output_dir / "invalid_smiles.csv"
    invalid_df.to_csv(invalid_output_csv, index=False)

    draw_top_fragments(
        result_df=result_df,
        output_path=output_dir / "top_fragments.png",
        top_n=top_n
    )

    print("分子片段统计完成")
    print("=" * 50)
    print(f"输入文件：{input_csv}")
    print(f"SMILES 列：{smiles_col}")
    print(f"原始分子数：{len(df)}")
    print(f"有效 SMILES 数：{len(df.dropna(subset=['canonical_smiles']))}")
    print(f"用于统计的分子数：{len(valid_df)}")
    print(f"无效 SMILES 数：{len(invalid_df)}")
    print(f"满足 min_mol_freq >= {min_mol_freq} 的片段数：{len(result_df)}")
    print("=" * 50)
    print(f"片段统计表：{output_csv}")
    print(f"高频片段图片：{output_dir / 'top_fragments.png'}")
    print(f"无效 SMILES 表：{invalid_output_csv}")


def draw_top_fragments(result_df, output_path, top_n=50):
    """
    绘制高频片段结构图。
    """
    if result_df is None or len(result_df) == 0:
        print("没有可绘制的高频片段。")
        return

    draw_df = result_df.head(top_n)

    mols = []
    legends = []

    for _, row in draw_df.iterrows():
        frag_smi = row["fragment_smiles"]
        mol = Chem.MolFromSmiles(frag_smi)

        if mol is None:
            continue

        mols.append(mol)

        legend = (
            f"Rank {row['rank']}\n"
            f"MolFreq={row['molecule_frequency']}\n"
            f"Occ={row['total_occurrence']}"
        )
        legends.append(legend)

    if len(mols) == 0:
        print("没有成功解析的片段结构，无法绘图。")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(260, 200),
        legends=legends
    )

    img.save(str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="统计 CSV 文件中的高频分子片段"
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
        default="fragment_frequency_results",
        help="输出文件夹"
    )

    parser.add_argument(
        "--min_mol_freq",
        type=int,
        default=2,
        help="片段至少出现在多少个分子中才输出，默认是 2"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="绘制前多少个高频片段，默认是 50"
    )

    parser.add_argument(
        "--keep_dummy_labels",
        action="store_true",
        help="是否保留 BRICS 连接位点类型，例如 [1*]、[3*]"
    )

    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="不对分子去重。如果打开，则重复 SMILES 会重复计数"
    )

    args = parser.parse_args()

    count_fragments(
        input_csv=args.input,
        smiles_col=args.smiles_col,
        output_dir=args.output_dir,
        min_mol_freq=args.min_mol_freq,
        top_n=args.top_n,
        keep_dummy_labels=args.keep_dummy_labels,
        deduplicate_molecules=not args.no_deduplicate
    )


if __name__ == "__main__":
    main()