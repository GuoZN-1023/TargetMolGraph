import pandas as pd
from pathlib import Path


# =========================
# 1. 文件路径设置
# =========================

AUGE_PATH = Path("AugE.xlsx")
PRIE_PATH = Path("PriE.csv")
WSE_PATH = Path("WSE.csv")

OUTPUT_AUGE_FILLED = Path("AugE_filled.csv")
OUTPUT_MERGED = Path("RE&WSE.csv")


# =========================
# 2. 读取文件
# =========================

auge = pd.read_excel(AUGE_PATH)
prie = pd.read_csv(PRIE_PATH)
wse = pd.read_csv(WSE_PATH)


# =========================
# 3. 清理列名
# =========================

auge.columns = auge.columns.str.strip()
prie.columns = prie.columns.str.strip()
wse.columns = wse.columns.str.strip()


# =========================
# 4. 检查必要列
# =========================

required_prie_cols = [
    "SMILES",
    "EP ID",
    "Binding Energy(eV)",
    "LUMO_sol(eV)",
    "HOMO_sol(eV)",
    "Dielectric constant of solvents",
]

if "SMILES" not in auge.columns:
    raise ValueError("AugE.xlsx 中缺少必要列：SMILES")

for col in required_prie_cols:
    if col not in prie.columns:
        raise ValueError(f"PriE.csv 中缺少必要列：{col}")


# =========================
# 5. 提取 PriE 中需要补充的性质
# =========================

prie_props = prie[
    [
        "SMILES",
        "EP ID",
        "Binding Energy(eV)",
        "LUMO_sol(eV)",
        "HOMO_sol(eV)",
        "Dielectric constant of solvents",
    ]
].copy()

prie_props = prie_props.drop_duplicates(subset=["SMILES"], keep="first")


# =========================
# 6. AugE 与 PriE 按 SMILES 匹配
# =========================

merged_auge = auge.merge(
    prie_props,
    on="SMILES",
    how="left",
    suffixes=("", "_from_PriE")
)


# =========================
# 7. 生成标准列
# =========================

standard_cols = [
    "EP ID",
    "Es-Ea (eV)",
    "LUMO_sol (eV)",
    "HOMO_sol (eV)",
    "Dielectric constant of solvents",
]

for col in standard_cols:
    if col not in merged_auge.columns:
        merged_auge[col] = pd.NA


# EP ID 补充
if "EP ID_from_PriE" in merged_auge.columns:
    merged_auge["EP ID"] = merged_auge["EP ID"].combine_first(
        merged_auge["EP ID_from_PriE"]
    )

# LUMO 补充
if "LUMO_sol(eV)" in merged_auge.columns:
    merged_auge["LUMO_sol (eV)"] = merged_auge["LUMO_sol (eV)"].combine_first(
        merged_auge["LUMO_sol(eV)"]
    )

# HOMO 补充
if "HOMO_sol(eV)" in merged_auge.columns:
    merged_auge["HOMO_sol (eV)"] = merged_auge["HOMO_sol (eV)"].combine_first(
        merged_auge["HOMO_sol(eV)"]
    )

# 介电常数补充
if "Dielectric constant of solvents_from_PriE" in merged_auge.columns:
    merged_auge["Dielectric constant of solvents"] = merged_auge[
        "Dielectric constant of solvents"
    ].combine_first(
        merged_auge["Dielectric constant of solvents_from_PriE"]
    )


# =========================
# 8. 计算 Es-Ea (eV)
# =========================

binding_col = "Binding Energy(eV)"
eps_col = "Dielectric constant of solvents"

merged_auge[binding_col] = pd.to_numeric(
    merged_auge[binding_col],
    errors="coerce"
)

merged_auge[eps_col] = pd.to_numeric(
    merged_auge[eps_col],
    errors="coerce"
)

calculated_es_ea = (
    merged_auge[binding_col]
    + 1.53 / merged_auge[eps_col]
    + 1.67
)

merged_auge["Es-Ea (eV)"] = merged_auge["Es-Ea (eV)"].combine_first(
    calculated_es_ea
)


# =========================
# 9. 删除旧列名和中间列
# =========================

drop_cols = [
    "EP ID_from_PriE",
    "Binding Energy(eV)",
    "LUMO_sol(eV)",
    "HOMO_sol(eV)",
    "Dielectric constant of solvents_from_PriE",
]

for col in drop_cols:
    if col in merged_auge.columns:
        merged_auge = merged_auge.drop(columns=col)


# =========================
# 10. 整理 AugE 列顺序
# =========================

final_columns = [
    "EP ID",
    "SMILES",
    "Es-Ea (eV)",
    "LUMO_sol (eV)",
    "HOMO_sol (eV)",
    "Dielectric constant of solvents",
]

for col in final_columns:
    if col not in merged_auge.columns:
        merged_auge[col] = pd.NA

merged_auge = merged_auge[final_columns]

merged_auge.to_csv(
    OUTPUT_AUGE_FILLED,
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 11. 整理 WSE 列名和列顺序
# =========================

# 如果 WSE 中还有旧格式列名，则映射为标准列名
if "LUMO_sol(eV)" in wse.columns and "LUMO_sol (eV)" not in wse.columns:
    wse["LUMO_sol (eV)"] = wse["LUMO_sol(eV)"]

if "HOMO_sol(eV)" in wse.columns and "HOMO_sol (eV)" not in wse.columns:
    wse["HOMO_sol (eV)"] = wse["HOMO_sol(eV)"]

# 删除旧格式列名
for col in ["LUMO_sol(eV)", "HOMO_sol(eV)"]:
    if col in wse.columns:
        wse = wse.drop(columns=col)

for col in final_columns:
    if col not in wse.columns:
        wse[col] = pd.NA

wse = wse[final_columns]


# =========================
# 12. 合并 AugE 与 WSE
# =========================

final_df = pd.concat(
    [merged_auge, wse],
    axis=0,
    ignore_index=True
)


# =========================
# 13. 去重
# =========================

final_df = final_df.drop_duplicates(keep="first")

# 如果你希望按 SMILES 去重，改用：
# final_df = final_df.drop_duplicates(subset=["SMILES"], keep="first")


# =========================
# 14. 最终列顺序整理并输出
# =========================

final_df = final_df[final_columns]

final_df.to_csv(
    OUTPUT_MERGED,
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 15. 输出检查信息
# =========================

print("任务完成！")
print(f"补充后的 AugE 文件：{OUTPUT_AUGE_FILLED}")
print(f"最终合并文件：{OUTPUT_MERGED}")
print()
print("AugE 原始行数：", len(auge))
print("补充后 AugE 行数：", len(merged_auge))
print("WSE 行数：", len(wse))
print("最终合并去重后行数：", len(final_df))

missing_rows = final_df[final_df[final_columns].isna().any(axis=1)]
print("最终文件中仍存在缺失值的行数：", len(missing_rows))