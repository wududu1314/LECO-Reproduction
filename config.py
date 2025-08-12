"""
LECO实验配置参数
严格遵循原论文中的所有超参数设置，并增加完善的风险控制机制
增加Windows兼容性和论文基准值验证
"""

import os
import sys
import psutil
import torch
import platform
import socket
from pathlib import Path

# Windows检测
IS_WINDOWS = platform.system() == "Windows"

class Config:
    # 三方法对比配置
    ENABLE_THREE_METHOD_COMPARISON = True  # 启用三方法对比
    BASELINE_METHOD = "complex_cot"  # 基线方法名称

    # Excel输出配置
    EXCEL_OUTPUT = True  # 启用Excel输出
    EXCEL_DETAILED = True  # 详细Excel报告

    # 日志配置 - 只显示关键信息
    LOG_LEVEL = "WARNING"  # 只显示警告和错误
    SHOW_PROGRESS_ONLY = True  # 只显示进度条

    # 新增：自定义测试配置
    CUSTOM_TEST_SIZE = 150
    MATH_SAMPLES_PER_TYPE = 30
    GSM8K_SAMPLES = 150
    ENABLE_PAPER_COMPARISON = False  # 禁用论文基准对比

    # 项目路径
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATASETS_DIR = PROJECT_ROOT / "datasets"
    PROMPTS_DIR = PROJECT_ROOT / "prompts"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

    # 模型配置
    MODEL_NAME = "deepseek-ai/deepseek-math-7b-rl"
    LOCAL_MODEL_PATH = str(MODELS_DIR / "deepseek-math-7b-rl")

    # 推理参数（与论文完全一致）
    MAX_NEW_TOKENS = 512  # GSM8K - 论文表1设置
    MAX_NEW_TOKENS_MATH = 1024  # MATH - 论文表2设置
    TEMPERATURE = 0.0  # 贪心解码
    NUM_BEAMS = 1  # 贪心解码
    DO_SAMPLE = False  # 贪心解码

    # LECO参数（与论文3.1节完全一致）
    MAX_ITERATIONS = 3  # 最大重思考轮数 - 论文算法1
    CONFIDENCE_K = 3  # 转换分数的token数量 - 论文公式3
    DIVERGENCE_TAU = 0.3  # KL散度温度参数 - 论文公式2
    DIVERGENCE_TYPE = "KL"  # 散度类型 - 论文使用KL散度

    # 改进策略参数（基于统计学理论，有充分理论支撑）
    STATISTICAL_THRESHOLD_COEFF = 1.5  # 对应86.6%置信区间，适合异常检测
    MIN_THRESHOLD = 0.05  # 下界保护，防止过度激进
    ANOMALY_DETECTION_MIN_SAMPLES = 3  # 异常检测最少样本数

    # 实验控制参数
    QUICK_TEST_SIZE = 10  # 快速测试样本数
    BATCH_SIZE = 50 if IS_WINDOWS else 250  # Windows降低批次大小
    RANDOM_SEED = 42  # 随机种子（确保可重现性）

    # 数据集配置
    GSM8K_TEST_PATH = str(DATASETS_DIR / "gsm8k" / "test.jsonl")
    MATH_TEST_PATH = str(DATASETS_DIR / "math" / "test")

    # 提示文件
    GSM8K_PROMPT_PATH = str(PROMPTS_DIR / "gsm8k_complex.txt")
    MATH_PROMPT_PATH = str(PROMPTS_DIR / "math_complex.txt")

    # GPU配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = "bfloat16"  # 与原论文实验设置一致
    ATTENTION_IMPL = None  # 禁用FlashAttention，使用默认注意力机制

    # 风险控制配置
    MAX_RETRIES = 3  # 最大重试次数
    TIMEOUT_SECONDS = 180 if IS_WINDOWS else 120  # Windows增加超时时间
    MEMORY_THRESHOLD = 0.8 if IS_WINDOWS else 0.85  # Windows更保守的内存阈值
    DISK_SPACE_THRESHOLD = 0.9  # 磁盘空间阈值

    # 容错机制
    ENABLE_CHECKPOINTS = True  # 启用断点续传
    CHECKPOINT_INTERVAL = 20 if IS_WINDOWS else 50  # Windows更频繁的检查点
    AUTO_MEMORY_CLEANUP = True  # 自动内存清理
    FALLBACK_TO_CPU = True  # GPU失败时自动切换CPU

    # 数据验证配置
    VALIDATE_INPUTS = True  # 验证输入数据
    VALIDATE_OUTPUTS = True  # 验证输出数据
    MAX_QUESTION_LENGTH = 2000  # 最大问题长度
    MAX_ANSWER_LENGTH = 100  # 最大答案长度

    # 实验重现性配置
    DETERMINISTIC_MODE = True  # 确定性模式
    RECORD_SYSTEM_INFO = True  # 记录系统信息
    RECORD_MODEL_INFO = True  # 记录模型信息

    # 论文基准值用于验证
    PAPER_BASELINES = {
        "deepseek-math-7b-rl": {
            "gsm8k_complex": 79.76,      # 论文Table 3
            "gsm8k_leco": 80.14,         # 期望改进到80.14
            "math_complex": 69.96,       # 论文Table 3
            "math_leco": 70.51           # 期望改进到70.51
        }
    }

    @classmethod
    def validate_environment(cls):
        """验证运行环境 - 增强版"""
        issues = []

        # Windows特殊处理
        if IS_WINDOWS:
            # 设置编码
            os.environ['PYTHONIOENCODING'] = 'utf-8'

            # 检查RDP会话
            try:
                import subprocess
                result = subprocess.run(['query', 'session'], capture_output=True, text=True)
                if 'rdp-tcp' in result.stdout.lower():
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info("检测到RDP会话，启用会话保护")
                    # 禁用屏幕超时
                    os.system('powercfg -change -monitor-timeout-ac 0')
            except:
                pass

        # 检查Python版本
        if sys.version_info < (3, 8):
            issues.append(f"Python版本过低: {sys.version_info}, 需要3.8+")

        # 检查内存
        memory = psutil.virtual_memory()
        min_memory = 12 if IS_WINDOWS else 16  # Windows降低要求
        if memory.total < min_memory * 1024**3:
            issues.append(f"系统内存不足: {memory.total/1024**3:.1f}GB, 推荐{min_memory}GB+")

        # 检查磁盘空间
        disk = psutil.disk_usage(str(cls.PROJECT_ROOT))
        free_gb = disk.free / 1024**3
        if free_gb < 50:  # 降低要求
            issues.append(f"磁盘空间不足: {free_gb:.1f}GB, 推荐50GB+")

        # 检查CUDA
        if cls.DEVICE == "cuda":
            if not torch.cuda.is_available():
                issues.append("CUDA不可用，将使用CPU（性能较慢）")
                cls.DEVICE = "cpu"
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                min_gpu_memory = 6 if IS_WINDOWS else 8  # Windows降低要求
                if gpu_memory < min_gpu_memory * 1024**3:
                    issues.append(f"GPU内存不足: {gpu_memory/1024**3:.1f}GB, 推荐{min_gpu_memory}GB+")

        # 检查必要目录
        required_dirs = [cls.DATASETS_DIR, cls.PROMPTS_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                issues.append(f"必要目录缺失: {dir_path}")

        return issues

    @classmethod
    def validate_results_against_paper(cls, our_accuracy: float, dataset: str):
        """验证结果与论文基准的一致性"""
        baseline_key = f"{dataset}_complex"
        if baseline_key in cls.PAPER_BASELINES["deepseek-math-7b-rl"]:
            baseline = cls.PAPER_BASELINES["deepseek-math-7b-rl"][baseline_key]
            diff = abs(our_accuracy - baseline)

            if diff > 2.0:  # 2%容差
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"结果偏差较大: 我们{our_accuracy:.2f}% vs 论文{baseline:.2f}%")
                return False
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"基准验证通过: {our_accuracy:.2f}% (论文: {baseline:.2f}%)")
                return True
        return True

    @classmethod
    def get_output_file(cls, dataset, method, test_type="full"):
        """生成输出文件路径"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}_{method}_{test_type}_{timestamp}.jsonl"
        return str(cls.RESULTS_DIR / filename)

    @classmethod
    def get_checkpoint_file(cls, dataset, method, batch_idx):
        """生成检查点文件路径"""
        filename = f"checkpoint_{dataset}_{method}_batch{batch_idx}.json"
        return str(cls.CHECKPOINTS_DIR / filename)

    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在"""
        dirs = [cls.RESULTS_DIR, cls.LOGS_DIR, cls.CHECKPOINTS_DIR]
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True, parents=True)

    @classmethod
    def set_random_seed(cls):
        """设置随机种子确保实验可重现"""
        import random
        import numpy as np

        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
            # 确保CUDA操作的确定性
            if cls.DETERMINISTIC_MODE:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    @classmethod
    def get_system_info(cls):
        """获取系统信息用于实验记录"""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "disk_free_gb": psutil.disk_usage(str(cls.PROJECT_ROOT)).free / 1024**3,
            "platform": platform.system(),
            "is_windows": IS_WINDOWS
        }

        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })

        return info

    @classmethod
    def check_resources(cls):
        """实时检查资源使用情况"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(cls.PROJECT_ROOT))
        
        warnings = []
        
        if memory.percent > cls.MEMORY_THRESHOLD * 100:
            warnings.append(f"内存使用率过高: {memory.percent:.1f}%")
        
        if disk.percent > cls.DISK_SPACE_THRESHOLD * 100:
            warnings.append(f"磁盘使用率过高: {disk.percent:.1f}%")
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory > cls.MEMORY_THRESHOLD:
                    warnings.append(f"GPU内存使用率过高: {gpu_memory:.1%}")
            except:
                pass  # 忽略GPU内存检查错误
        
        return warnings