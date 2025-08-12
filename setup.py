#!/usr/bin/env python3
"""
LECO复现项目一次性环境配置脚本（修复版）
跳过conda环境创建，避免Windows权限问题
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
import logging
import socket
import platform
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LECOSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.models_dir = self.project_root / "models"
        self.datasets_dir = self.project_root / "datasets"
        self.prompts_dir = self.project_root / "prompts"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "logs"
        self.core_dir = self.project_root / "core"
        self.checkpoints_dir = self.project_root / "checkpoints"

    def run_cmd(self, cmd, description="", critical=True):
        """运行CMD命令并处理错误"""
        logger.info(f"执行: {description}")
        logger.debug(f"命令: {cmd}")

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error(f"命令执行失败: {result.stderr}")
                if critical:
                    raise RuntimeError(f"关键步骤失败: {description}")
                return False
            logger.info(f"成功: {description}")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"命令超时: {description}")
            if critical:
                raise RuntimeError(f"命令超时: {description}")
            return False
        except Exception as e:
            logger.error(f"执行异常: {e}")
            if critical:
                raise RuntimeError(f"执行异常: {description}")
            return False

    def create_directories(self):
        """创建项目目录结构"""
        logger.info("创建项目目录结构")
        dirs = [
            self.models_dir, self.datasets_dir, self.prompts_dir,
            self.results_dir, self.logs_dir, self.core_dir, self.checkpoints_dir,
            self.datasets_dir / "gsm8k", self.datasets_dir / "math"
        ]

        for dir_path in dirs:
            try:
                dir_path.mkdir(exist_ok=True, parents=True)
                logger.debug(f"创建目录: {dir_path}")
            except Exception as e:
                logger.error(f"创建目录失败 {dir_path}: {e}")
                raise RuntimeError(f"目录创建失败: {dir_path}")

        logger.info("目录结构创建完成")

    def install_dependencies(self):
        """安装Python依赖包"""
        logger.info("安装Python依赖包（GPU版本）")

        # Windows特殊处理
        if platform.system() == "Windows":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            logger.info("Windows环境变量已设置")

        # 配置pip镜像源
        self.run_cmd("pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/",
                     "配置pip镜像源", critical=False)

        # 优先尝试conda安装PyTorch GPU版本
        logger.info("尝试使用conda安装PyTorch GPU版本")
        conda_pytorch_commands = [
            "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y",
            "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y",
        ]

        pytorch_installed = False
        for cmd in conda_pytorch_commands:
            if self.run_cmd(cmd, f"conda安装PyTorch: {cmd}", critical=False):
                logger.info("conda安装PyTorch成功")
                pytorch_installed = True
                break
            else:
                logger.warning(f"conda命令失败: {cmd}")

        # 如果conda失败，使用pip安装GPU版本
        if not pytorch_installed:
            logger.info("conda安装失败，尝试pip安装PyTorch GPU版本")
            pip_pytorch_commands = [
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --timeout 1800",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --timeout 1800",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --timeout 1800",
            ]

            for cmd in pip_pytorch_commands:
                if self.run_cmd(cmd, f"pip安装PyTorch: {cmd}", critical=False):
                    logger.info("pip安装PyTorch GPU版本成功")
                    pytorch_installed = True
                    break
                else:
                    logger.warning(f"pip命令失败: {cmd}")

        # 最后的备选方案：安装CPU版本（仅作为最后手段）
        if not pytorch_installed:
            logger.warning("所有GPU版本PyTorch安装失败，安装CPU版本作为备选")
            cpu_cmd = "pip install torch torchvision torchaudio --timeout 1800"
            if not self.run_cmd(cpu_cmd, "安装CPU版本PyTorch", critical=True):
                logger.error("连CPU版本PyTorch也安装失败")
                return False
            else:
                logger.warning("已安装CPU版本PyTorch，性能会受影响")

        # 验证PyTorch安装和GPU可用性
        logger.info("验证PyTorch安装...")
        verification_code = '''
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print("✅ GPU版本PyTorch安装成功")
    else:
        print("⚠️  CPU版本PyTorch，性能会受影响")
    '''

        verification_cmd = f'python -c "{verification_code}"'
        self.run_cmd(verification_cmd, "验证PyTorch安装", critical=False)

        # 安装其他依赖包
        logger.info("安装其他依赖包...")
        other_deps = [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "pandas>=1.3.0",
            "tqdm>=4.62.0",
            "datasets>=2.0.0",
            "jsonlines>=3.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "psutil>=5.8.0"
            "xlsxwriter>=3.0.0"
        ]

        # 批量安装其他包
        pip_cmd = f"pip install {' '.join(other_deps)} --timeout 1800"
        if not self.run_cmd(pip_cmd, "批量安装其他依赖包", critical=False):
            # 逐个安装
            logger.info("批量安装失败，逐个安装")
            failed_deps = []
            for dep in other_deps:
                if not self.run_cmd(f"pip install {dep} --timeout 1800", f"安装{dep}", critical=False):
                    failed_deps.append(dep)

            if failed_deps:
                logger.error(f"以下依赖安装失败: {failed_deps}")
                logger.info("请尝试手动安装或检查网络连接")
                return False

        logger.info("所有依赖安装完成")

        # 最终GPU检查
        logger.info("执行最终GPU环境检查...")
        final_check_code = '''
    import torch
    import platform
    print("="*50)
    print("LECO环境配置完成检查")
    print("="*50)
    print(f"操作系统: {platform.system()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {name} ({memory:.1f}GB)")
        print("🚀 GPU环境配置成功，可以进行高速推理！")
    else:
        print("⚠️  仅CPU可用，推理速度会较慢")
    print("="*50)
    '''

        final_check_cmd = f'python -c "{final_check_code}"'
        self.run_cmd(final_check_cmd, "最终环境检查", critical=False)

        return True

    def download_model(self):
        """配置模型下载"""
        logger.info("配置模型下载")

        # 设置HuggingFace镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 创建模型下载脚本
        download_script = '''
import os
import logging
import platform
import socket
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows特殊处理
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(300)  # 5分钟超时

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_name = "deepseek-ai/deepseek-math-7b-rl"
save_path = "./models/deepseek-math-7b-rl"

logger.info("开始下载DeepSeek模型")
logger.info("这可能需要较长时间，请耐心等待")

try:
    # 下载tokenizer
    logger.info("下载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    logger.info("Tokenizer下载完成")

    # 下载模型
    logger.info("下载模型权重")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.save_pretrained(save_path)
    logger.info("模型下载完成")

    # 验证模型完整性
    logger.info("验证模型文件")
    config_path = os.path.join(save_path, "config.json")
    if os.path.exists(config_path):
        logger.info("模型配置文件验证通过")
    else:
        raise Exception("模型配置文件缺失")

    logger.info("模型下载和验证成功完成")

    # Windows: 恢复默认超时
    if platform.system() == "Windows":
        socket.setdefaulttimeout(original_timeout)

except Exception as e:
    logger.error(f"下载失败: {e}")
    logger.error("请检查网络连接或手动下载模型")
    raise
'''

        try:
            with open("download_model.py", "w", encoding="utf-8") as f:
                f.write(download_script)
            logger.info("模型下载脚本已创建: download_model.py")
        except Exception as e:
            logger.error(f"创建下载脚本失败: {e}")
            raise RuntimeError("模型下载脚本创建失败")

    def create_prompt_files(self):
        """创建提示文件 - 与原论文完全一致"""
        logger.info("创建提示文件")

        # GSM8K提示
        gsm8k_prompt = """Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Answer: Let's think step by step
Step1: Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
Step2: For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Step3: Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
Step4: However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
Step5: They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
Step6: And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
Step7: So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
Step8: They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
Step9: They will need to plan to study 4 days to allow for all the time they need.
Step10: The answer is \\boxed{4}

Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Answer: Let's think step by step
Step1: Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
Step2: His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
Step3: They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
Step4: All together his team scored 50+24+10= 84 points
Step5: Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
Step6: His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
Step7: They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
Step8: All together Mark's opponents scored 100+12+5=117 points
Step9: The total score for the game is both team's scores added together, so it is 84+117=201 points
Step10: The answer is \\boxed{201}"""

        # MATH提示
        math_prompt = """Question: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.
A: Let's think step by step
Step1: Kevin hops $1/3$ of the remaining distance with every hop.
Step2: His first hop takes $1/3$ closer.
Step3: For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.
Step4: For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.
Step5: In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.
Step6: We want to find how far he has hopped after five hops.
Step7: This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.
Step8: Thus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.
Step9: The answer is \\boxed{\\frac{211}{243}}

Question: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?
A: Let's think step by step
Step1: We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,
Step2: resulting in  $(x+7)^2-49 + (y-2)^2-4=10$,
Step3: or $(x+7)^2+(y-2)^2=63$.
Step4: This is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$
Step5: so the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.
Step6: The answer is \\boxed{63\\pi}"""

        try:
            with open(self.prompts_dir / "gsm8k_complex.txt", "w", encoding="utf-8") as f:
                f.write(gsm8k_prompt)

            with open(self.prompts_dir / "math_complex.txt", "w", encoding="utf-8") as f:
                f.write(math_prompt)

            logger.info("提示文件创建完成")
        except Exception as e:
            logger.error(f"创建提示文件失败: {e}")
            raise RuntimeError("提示文件创建失败")

    def validate_environment(self):
        """验证环境配置"""
        logger.info("验证环境配置")

        # 检查Python版本
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            logger.error(f"Python版本不符合要求: {python_version}，需要Python 3.8+")
            return False

        # 检查CUDA可用性
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                logger.info(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
            else:
                logger.warning("CUDA不可用，将使用CPU推理（速度较慢）")
        except ImportError:
            logger.warning("PyTorch未安装，跳过CUDA检查")

        logger.info("环境验证完成")
        return True

    def run_setup(self):
        """运行完整安装流程"""
        logger.info("LECO项目环境配置开始")

        try:
            # 1. 创建目录结构
            self.create_directories()

            # 2. 检查参数
            if len(sys.argv) > 1 and sys.argv[1] == "--install-deps":
                # 安装依赖
                if not self.install_dependencies():
                    return False
                if not self.validate_environment():
                    return False
                logger.info("依赖安装完成")
                return True

            # 3. 创建提示文件
            self.create_prompt_files()

            # 4. 下载模型脚本
            self.download_model()

            logger.info("环境配置完成")
            logger.info("下一步操作:")
            logger.info("1. 手动下载数据集:")
            logger.info(
                "   - GSM8K: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl")
            logger.info("   - 保存到: datasets/gsm8k/test.jsonl")
            logger.info("   - MATH: https://people.eecs.berkeley.edu/~hendrycks/MATH.tar")
            logger.info("   - 解压到: datasets/math/test/")
            logger.info("2. 运行: python setup.py --install-deps (安装依赖)")
            logger.info("3. 运行: python download_model.py (下载模型)")
            logger.info("4. 创建核心模块文件 (手动复制粘贴代码)")
            logger.info("5. 运行实验: python run_experiment.py")

            return True

        except Exception as e:
            logger.error(f"环境配置失败: {e}")
            return False


if __name__ == "__main__":
    setup = LECOSetup()
    success = setup.run_setup()
    if not success:
        sys.exit(1)