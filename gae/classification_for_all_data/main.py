# # # main.py
# # from train import train_and_evaluate
# # import datetime 
# # if __name__ == "__main__":
# #     # Define paths
# #     folder_path = "./data/calcium_data_all/GABA_all/day45_GABA"  # Replace with your folder
# #     model_save_path = "/home/featurize/work/ylx/MEA/gae/xgboost_model.json"

# #     # Train and evaluate the model
# #     train_and_evaluate(folder_path, model_save_path)

# from train import train_and_evaluate
# import os
# from pathlib import Path
# from datetime import datetime
# import pytz
# import logging
# # def get_datasets(base_path):
# #     """获取数据集目录下的所有子文件夹名称"""
# #     return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
# def setup_logging(result_dir):
#     """配置日志系统"""
#     log_file = result_dir / "training.log"
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),  # 保存到文件
#             logging.StreamHandler()        # 同时输出到控制台
#         ]
#     )
# if __name__ == "__main__":
#     # 基础路径（包含所有数据集的目录）
#     base_data_path = "../../data/calcium_data_all/"  # 所有数据集在此目录下
#     china_timezone = pytz.timezone("Asia/Shanghai")
#     # 获取所有数据集名称（自动扫描子目录）
#     # datasets = get_datasets(base_data_path)
#     # 或者手动指定数据集列表：
#     datasets = ["cnqx_apv", "GABA", "glu", "sac", "sr", 
#                 "cnqx_apv_all/day90_cnqx_apv", "cnqx_apv_all/day120_cnqx_apv",
#                 "GABA_all/day45_GABA", "GABA_all/day90_GABA", "GABA_all/day120_GABA", 
#                 "glu_all/day45_glu", "glu_all/day90_glu", "glu_all/day120_glu",
#                 "sac_all/day90_sac", "sac_all/day120_sac",
#                 "sr_all/day90_sr", "sr_all/day120_sr"]  

#     for dataset in datasets:
#         # 动态生成路径
#         folder_path = os.path.join(base_data_path, dataset)
        
#         # 创建带时间戳的结果目录（格式：数据集名称-年月日_时分秒）
#         timestamp = datetime.now(china_timezone).strftime("%Y-%m-%d_%H-%M-%S")
#         dataset_name = dataset.replace("/", "-")
#         result_dir = Path(f"./results/{dataset_name}-{timestamp}")
#         result_dir.mkdir(parents=True, exist_ok=True)
#         setup_logging(result_dir)
#         # 动态模型保存路径（每个数据集单独保存）
#         # model_save_path = result_dir / "xgboost_model.json"
#         paths = {
#             "data_path": folder_path,
#             "result_dir": result_dir,  # 整个结果目录
#             "embedding_plot_path": result_dir / "embedding_plot.png",
#             "confusion_matrix_path": result_dir / "confusion_matrix.png"
#             # 注意：不包含model_save_path
#         }

#         logging.info(f"🔍 正在处理数据集: {dataset}")
#         try:
        
#             # 调用原有训练函数
#             train_and_evaluate(paths)
            
#             logging.info(f"✅ 完成处理: {dataset}\n")
#         except Exception as e:
#             logging.error(f"训练失败: {str(e)}", exc_info=True)


from train import train_and_evaluate
import os
from pathlib import Path
from datetime import datetime
import pytz
import logging
import json

china_timezone = pytz.timezone("Asia/Shanghai")
def setup_experiment_logging(exp_dir):
    """配置实验级别的日志系统（整个exp共用一个log文件）"""
    log_file = exp_dir / "experiment.log"
    
    # 清除之前的配置
    logging.root.handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

def create_experiment_dir(base_dir="../../results"):
    """创建实验目录（exp1, exp2,...）"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # 查找已存在的实验目录
    existing_exps = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("exp")]
    next_num = len(existing_exps) + 1
    
    exp_dir = base_path / f"exp{next_num}"
    exp_dir.mkdir()
    return exp_dir

if __name__ == "__main__":
    # 1. 创建实验目录
    exp_dir = create_experiment_dir()
    setup_experiment_logging(exp_dir)
    
    logging.info(f"🎯 开始新实验：{exp_dir.name}")
    logging.info(f"⏰ 实验开始时间：{datetime.now(china_timezone).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. 数据集配置
    base_data_path = Path("../../data/calcium_data_all/")
    china_timezone = pytz.timezone("Asia/Shanghai")
    
    datasets = [
        "cnqx_apv", "GABA", "glu", "sac", "sr",
        "cnqx_apv_all/day90_cnqx_apv", "cnqx_apv_all/day120_cnqx_apv",
        "GABA_all/day45_GABA", "GABA_all/day90_GABA", "GABA_all/day120_GABA",
        "glu_all/day45_glu", "glu_all/day90_glu", "glu_all/day120_glu",
        "sac_all/day90_sac", "sac_all/day120_sac",
        "sr_all/day90_sr", "sr_all/day120_sr"
    ]

    # 3. 处理每个数据集
    for dataset in datasets:
        dataset_name = dataset.replace("/", "-")
        dataset_dir = exp_dir / dataset_name
        dataset_dir.mkdir()
        
        # 创建子目录结构
        (dataset_dir / "embeddings").mkdir()
        (dataset_dir / "models").mkdir()
        (dataset_dir / "figures").mkdir()
        
        paths = {
            "data_path": base_data_path / dataset,
            "result_dir": dataset_dir,
            "embedding_plot_path": dataset_dir / "figures/embeddings.png",
            "confusion_matrix_path": dataset_dir / "figures/confusion_matrix.png",
            "model_save_path": dataset_dir / "models/xgboost_model.json",
            "loss_path": dataset_dir / "figures/loss_curve.png"
        }

        logging.info(f"\n🔍 开始处理数据集: {dataset}")
        logging.info(f"📁 数据路径: {paths['data_path']}")
        
        try:
            # 4. 训练和评估
            metrics = train_and_evaluate(paths)
            
            # 保存指标
            with open(dataset_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
                
            logging.info(f"✅ 完成 {dataset} | 准确率: {metrics['accuracy']:.2f}")
            
        except Exception as e:
            logging.error(f"❌ {dataset} 处理失败: {str(e)}", exc_info=True)
            # 记录失败状态
            with open(dataset_dir / "FAILED", "w") as f:
                f.write(str(e))
    
    # 5. 实验总结
    success_count = sum(1 for d in exp_dir.iterdir() if d.is_dir() and not (d / "FAILED").exists())
    logging.info(f"\n🎉 实验完成 {success_count}/{len(datasets)} 个数据集成功")
    logging.info(f"📂 实验结果目录: {exp_dir.absolute()}")