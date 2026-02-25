import os
from PIL import Image
from torch.utils.data import Dataset

class FinancialCorpusDataset(Dataset):
    """
    全量文档池数据集，用于提取所有候选页面的视觉特征。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 扫描目录下所有的 .jpg 图像
        self.image_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.lower().endswith('.jpg')
        ])
        
        if not self.image_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到 .jpg 图像")
            
        print(f"[CORPUS] 扫描完毕，发现 {len(self.image_files)} 张财报页面图像。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.data_dir, filename)
        
        # 获取 page_id (假设文件名即为 page_id)
        page_id = os.path.splitext(filename)[0]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] 无法读取图片 {img_path}: {e}")
            image = Image.new('RGB', (1024, 1024), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'page_id': page_id
        }
