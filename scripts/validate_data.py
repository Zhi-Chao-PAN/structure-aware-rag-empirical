import os
import sys
from pathlib import Path
import pandas as pd

def validate_data(data_dir="data/vision_rag"):
    print(f"=== 验证数据完整性 ({data_dir}) ===")
    data_path = Path(data_dir)
    
    # 1. Check parquet
    parquet_file = data_path / "page_metadata.parquet"
    if not parquet_file.exists():
        print(f"❌ [FAIl] 缺少 page_metadata.parquet: {parquet_file}")
        return False
    
    try:
        df = pd.read_parquet(parquet_file)
        if len(df) == 0:
            print(f"❌ [FAIL] page_metadata.parquet 为空")
            return False
        print(f"✅ [PASS] page_metadata.parquet 包含 {len(df)} 行数据")
    except Exception as e:
        print(f"❌ [FAIL] 读取 page_metadata.parquet 失败: {e}")
        return False

    # 2. Check images directory
    img_dir = data_path / "page_images"
    if not img_dir.exists():
        print(f"❌ [FAIL] 缺少 page_images 目录")
        return False
        
    img_files = list(img_dir.glob("*.jpg"))
    if len(img_files) != len(df):
        print(f"⚠️ [WARN] 图像数量 ({len(img_files)}) 与 parquet 行数 ({len(df)}) 不一致")
    else:
        print(f"✅ [PASS] 找到 {len(img_files)} 张页面图像")

    # 3. Check QA file
    qa_files = [data_path / "visfinqa" / "test.json", data_path / "oracle_qa.json"]
    found_qa = any(f.exists() for f in qa_files)
    if not found_qa:
        print(f"❌ [FAIL] 缺少评测标注文件 (在 visfinqa/test.json 或 oracle_qa.json)")
        return False
    print(f"✅ [PASS] 找到评测标注文件")

    # 4. Check spatial masks
    mask_dir = data_path / "spatial_masks"
    if not mask_dir.exists():
        print(f"❌ [FAIL] 缺少 spatial_masks 目录")
        return False
    
    unique_tickers = df["ticker"].unique()
    mask_files = list(mask_dir.glob("*.pt"))
    if len(mask_files) < len(unique_tickers):
        print(f"⚠️ [WARN] spatial_masks 对应文件数 ({len(mask_files)}) 少于 unique tickers ({len(unique_tickers)})")
    else:
        print(f"✅ [PASS] 找到 {len(mask_files)} 个 ticker 对应的 mask 分片")

    print("\n=== 数据验证完成 ===")
    return True

if __name__ == "__main__":
    success = validate_data()
    sys.exit(0 if success else 1)
