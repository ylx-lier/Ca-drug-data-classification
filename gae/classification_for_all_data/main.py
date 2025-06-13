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
    
    logging.info("开始联合训练实验...")
    
    # 定义路径
    paths = {
        "result_dir": exp_dir,
        "embedding_plot_path": exp_dir / "figures/embeddings.png",
        "confusion_matrix_path": exp_dir / "figures/confusion_matrix.png",
        "model_save_path": exp_dir / "models/xgboost_model.json",
        "loss_path": exp_dir / "figures/loss_curve.png",
        "tensorboard_path": exp_dir / "tensorboard"
    }
    for path in paths.values():
        dir_path = path.parent  # 取到目录名，比如 exp_dir/figures
        dir_path.mkdir(parents=True, exist_ok=True)  # 如果不存在就创建，多层也没关系
    # 执行联合训练和评估
    results = train_and_evaluate(paths)
    
    # 保存所有结果
    with open(exp_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info("实验完成！")