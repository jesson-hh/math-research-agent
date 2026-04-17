"""
Industry -> sector mapping for CSI300 / A-share stocks.

We collapse the ~100 fine-grained tushare industries into 11 broad sectors
(plus UNKNOWN) for use as a low-dimensional factor conditioning signal.

Source: G:/stocks/stock_data/parquet/tushare_stock_basic.parquet
Format: ts_code='000001.SZ' -> qlib-style 'SZ000001'
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


SECTOR_NAMES = [
    "UNKNOWN",      # 0 — unmatched / missing industry
    "FINANCE",      # 1
    "TECH",         # 2
    "MEDIA",        # 3
    "HEALTHCARE",   # 4
    "CONSUMER",     # 5
    "INDUSTRIAL",   # 6
    "MATERIALS",    # 7
    "ENERGY",       # 8
    "METALS",       # 9
    "REAL_ESTATE",  # 10
    "TRANSPORT",    # 11
]

N_SECTORS = len(SECTOR_NAMES)  # 12


# Chinese industry name -> sector id
INDUSTRY_TO_SECTOR: dict[str, int] = {
    # 金融 -> FINANCE
    "证券": 1, "银行": 1, "保险": 1, "多元金融": 1,
    # TMT / 科技 -> TECH
    "元器件": 2, "软件服务": 2, "半导体": 2, "互联网": 2,
    "通信设备": 2, "IT设备": 2, "电信运营": 2, "电器仪表": 2,
    # 传媒 -> MEDIA
    "影视音像": 3, "广告包装": 3, "出版业": 3, "文教休闲": 3,
    # 医药 -> HEALTHCARE
    "化学制药": 4, "中成药": 4, "医疗保健": 4,
    "生物制药": 4, "医药商业": 4,
    # 消费 -> CONSUMER
    "家用电器": 5, "白酒": 5, "食品": 5, "乳制品": 5,
    "啤酒": 5, "红黄酒": 5, "软饮料": 5, "服饰": 5,
    "百货": 5, "超市连锁": 5, "商品城": 5, "电器连锁": 5,
    "家居用品": 5, "日用化工": 5, "旅游服务": 5,
    "酒店餐饮": 5, "旅游景点": 5,
    "农业综合": 5, "饲料": 5, "种植业": 5,
    "商贸代理": 5, "综合类": 5, "其他商业": 5,
    # 工业/制造 -> INDUSTRIAL
    "电气设备": 6, "建筑工程": 6, "专用机械": 6, "工程机械": 6,
    "运输设备": 6, "汽车配件": 6, "汽车整车": 6, "船舶": 6,
    "机械基件": 6, "化工机械": 6, "装修装饰": 6, "新型电力": 6,
    # 材料/化工 -> MATERIALS
    "化工原料": 7, "化纤": 7, "农药化肥": 7, "塑料": 7,
    "橡胶": 7, "染料涂料": 7, "矿物制品": 7, "玻璃": 7,
    "水泥": 7, "其他建材": 7,
    # 能源 -> ENERGY
    "火力发电": 8, "水力发电": 8, "煤炭开采": 8, "石油开采": 8,
    "石油加工": 8, "焦炭加工": 8, "供气供热": 8,
    # 金属 -> METALS
    "小金属": 9, "铜": 9, "普钢": 9, "铝": 9,
    "黄金": 9, "铅锌": 9, "特种钢": 9, "钢加工": 9,
    # 地产/基建 -> REAL_ESTATE
    "全国地产": 10, "区域地产": 10, "园区开发": 10,
    "水务": 10, "环境保护": 10,
    # 交运 -> TRANSPORT
    "航空": 11, "仓储物流": 11, "空运": 11, "港口": 11,
    "水运": 11, "铁路": 11, "机场": 11, "路桥": 11,
}


def ts_to_qlib_code(ts_code: str) -> str:
    """'000001.SZ' -> 'SZ000001'."""
    if "." not in ts_code:
        return ts_code
    num, ex = ts_code.split(".")
    return f"{ex}{num}"


def load_industry_df(parquet_path: str | Path) -> pd.DataFrame:
    """Load tushare stock_basic and add 'qlib' column + 'sector' column."""
    df = pd.read_parquet(parquet_path)
    df = df[["ts_code", "industry"]].copy()
    df["qlib"] = df["ts_code"].apply(ts_to_qlib_code)
    df["sector"] = df["industry"].map(INDUSTRY_TO_SECTOR).fillna(0).astype(int)
    return df[["qlib", "industry", "sector"]]


def build_sector_labels(
    qlib_codes: list[str] | np.ndarray,
    parquet_path: str | Path = "G:/stocks/stock_data/parquet/tushare_stock_basic.parquet",
) -> np.ndarray:
    """
    Given an array of qlib codes (e.g. from CSI300 panel npz), return a
    (N_stocks,) int array of sector ids. Unmatched codes get sector 0 (UNKNOWN).
    """
    df = load_industry_df(parquet_path)
    lookup = dict(zip(df["qlib"], df["sector"]))
    labels = np.array([lookup.get(c, 0) for c in qlib_codes], dtype=np.int64)
    return labels


def summary(labels: np.ndarray) -> dict:
    """Per-sector stock counts."""
    counts = np.bincount(labels, minlength=N_SECTORS)
    return {SECTOR_NAMES[i]: int(counts[i]) for i in range(N_SECTORS)}


if __name__ == "__main__":
    # Smoke test: load CSI300 codes and show sector distribution
    import sys
    if len(sys.argv) < 2:
        npz_path = "data/csi300_2015_2024.npz"
    else:
        npz_path = sys.argv[1]
    d = np.load(npz_path, allow_pickle=True)
    codes = d["codes"]
    labels = build_sector_labels(codes)
    print(f"N stocks: {len(labels)}")
    print("Per-sector counts:")
    for name, n in summary(labels).items():
        print(f"  {n:4d}  {name}")
