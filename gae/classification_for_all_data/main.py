#!/usr/bin/env python3
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
    name_exp = "all_data"
    exp_dir = base_path / f"exp{next_num}"
    exp_dir.mkdir()
    return exp_dir

if __name__ == "__main__":
    # 创建实验目录
    exp_dir = create_experiment_dir()
    setup_experiment_logging(exp_dir)
    
    # 模型选择 - 可以在这里修改
    # 可选: "simple", "graphmae", "original"
    model_type = "simple"  # 默认使用simple模型，推荐！
    
    logging.info(f"开始联合训练实验，使用模型: {model_type.upper()}")
    
    # 定义路径
    paths = {
        "result_dir": exp_dir,
        "embedding_plot_path": exp_dir / "figures/embeddings.png",
        "confusion_matrix_path": exp_dir / "figures/confusion_matrix.png",
        "model_save_path": exp_dir / "models/xgboost_model.json",
        "loss_path": exp_dir / "figures/loss_curve.png",
        "tensorboard_path": exp_dir / "tensorboard"
    }
    
    # 创建所有必要的目录
    for path in paths.values():
        if isinstance(path, Path):
            if path.suffix:  # 如果是文件路径
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # 如果是目录路径
                path.mkdir(parents=True, exist_ok=True)
    
    # 执行联合训练和评估
    results = train_and_evaluate(paths, model_type=model_type)
    
    # 保存所有结果
    results_with_config = {
        "model_type": model_type,
        "experiment_config": {
            "model": model_type,
            "timestamp": datetime.now(china_timezone).isoformat()
        },
        "results": results
    }
    
    with open(exp_dir / "all_results.json", "w") as f:
        json.dump(results_with_config, f, indent=2)
    
    logging.info("实验完成！")