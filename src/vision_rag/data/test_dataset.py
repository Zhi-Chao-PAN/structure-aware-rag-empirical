import json
import os
from PIL import Image
from torch.utils.data import Dataset

class FinancialTestDataset(Dataset):
    def __init__(self, data_dir, annotation_file=None, transform=None):
        """
        data_dir: 存放 SEC 财报图片的文件夹路径
        annotation_file: 包含 Query 和对应正确财报页码的 JSON 文件路径
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 默认回退路径
        if annotation_file is None:
            # 兼容 A800 目录结构
            annotation_file = "/root/autodl-tmp/project/data/vision_rag/oracle_qa.json"
            if not os.path.exists(annotation_file):
               # 本地回退测试路径
               annotation_file = "data/vision_rag/oracle_qa.json"

        # 加载测试集的问答对
        print(f"[DATA] 正在读取标注文件: {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"[DATA] 成功加载 {len(self.data)} 条真实评测数据！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # JSON结构为: "page_id": "AAPL_FY2022_Q1_10-Q_p017", "query": "..."
        img_filename = item['page_id'] + ".jpg"
        query_text = item['query']
        
        # 读取图片
        img_path = os.path.join(self.data_dir, img_filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] 无法读取图片 {img_path}: {e}")
            # 如果某张图坏了，可以用全白图像占位防止崩溃
            image = Image.new('RGB', (1024, 1024), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'query': query_text,
            'doc_id': item['page_id']  # 用于后续评测算 nDCG 时核对 Ground Truth
        }
