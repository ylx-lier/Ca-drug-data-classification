# # main.py
# from train import train_and_evaluate
# import datetime 
# if __name__ == "__main__":
#     # Define paths
#     folder_path = "./data/calcium_data_all/GABA_all/day45_GABA"  # Replace with your folder
#     model_save_path = "/home/featurize/work/ylx/MEA/gae/xgboost_model.json"

#     # Train and evaluate the model
#     train_and_evaluate(folder_path, model_save_path)

from train import train_and_evaluate
import os
from pathlib import Path
from datetime import datetime
import pytz

def get_datasets(base_path):
    """获取数据集目录下的所有子文件夹名称"""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

if __name__ == "__main__":
    # 基础路径（包含所有数据集的目录）
    base_data_path = "../../data/calcium_data_all/"  # 所有数据集在此目录下
    china_timezone = pytz.timezone("Asia/Shanghai")
    # 获取所有数据集名称（自动扫描子目录）
    # datasets = get_datasets(base_data_path)
    # 或者手动指定数据集列表：
    datasets = ["cnqx_apv", "GABA", "glu", "sac", "sr", 
                "cnqx_apv_all/day90_cnqx_apv", "cnqx_apv_all/day120_cnqx_apv",
                "GABA_all/day45_GABA", "GABA_all/day90_GABA", "GABA_all/day120_GABA", 
                "glu_all/day45_glu", "glu_all/day90_glu", "glu_all/day120_glu",
                "sac_all/day90_sac", "sac_all/day120_sac",
                "sr_all/day90_sr", "sr_all/day120_sr"]  

    for dataset in datasets:
        # 动态生成路径
        folder_path = os.path.join(base_data_path, dataset)
        
        # 创建带时间戳的结果目录（格式：数据集名称-年月日_时分秒）
        timestamp = datetime.now(china_timezone).strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = dataset.replace("/", "-")
        result_dir = Path(f"./results/{dataset_name}-{timestamp}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 动态模型保存路径（每个数据集单独保存）
        # model_save_path = result_dir / "xgboost_model.json"
        paths = {
            "data_path": folder_path,
            "result_dir": result_dir,  # 整个结果目录
            "embedding_plot_path": result_dir / "embedding_plot.png",
            "confusion_matrix_path": result_dir / "confusion_matrix.png"
            # 注意：不包含model_save_path
        }

        print(f"🔍 正在处理数据集: {dataset}")
    
        
        # 调用原有训练函数
        train_and_evaluate(paths)
        
        print(f"✅ 完成处理: {dataset}\n")