from train import train_and_evaluate
import os
from pathlib import Path
from datetime import datetime
import pytz
import logging
import json
from tqdm import tqdm
from logging import StreamHandler

china_timezone = pytz.timezone("Asia/Shanghai")

class ChinaTimeFormatter(logging.Formatter):
    """自定义Formatter，使用中国时区"""
    converter = lambda *args: datetime.now(china_timezone).timetuple()

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, china_timezone)
        if datefmt:
            return ct.strftime(datefmt)
        else:
            return ct.strftime("%Y-%m-%d %H:%M:%S")

class TqdmLoggingHandler(StreamHandler):
    """特殊的logging handler用于与tqdm兼容"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_experiment_logging(exp_dir):
    """配置实验级别的日志系统（整个exp共用一个log文件）"""
    log_file = exp_dir / "experiment.log"
    
    # 清除之前的配置
    logging.root.handlers = []
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建中国时区的formatter
    formatter = ChinaTimeFormatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台handler (tqdm兼容)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
    
    datasets = [
        "cnqx_apv", "GABA", "glu", "sac", "sr",
        "cnqx_apv_all/day90_cnqx_apv", "cnqx_apv_all/day120_cnqx_apv",
        "GABA_all/day45_GABA", "GABA_all/day90_GABA", "GABA_all/day120_GABA",
        "glu_all/day45_glu", "glu_all/day90_glu", "glu_all/day120_glu",
        "sac_all/day90_sac", "sac_all/day120_sac",
        "sr_all/day90_sr", "sr_all/day120_sr"
    ]

    # 3. 使用tqdm包装数据集循环
    for dataset in tqdm(datasets, desc="处理数据集中"):
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