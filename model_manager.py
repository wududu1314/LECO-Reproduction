"""
模型管理和推理模块
实现与原论文完全一致的模型推理过程，增强Windows兼容性和错误恢复
"""

import torch
import gc
import time
import logging
import platform
import socket
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        self.retry_count = 0
        self._generation_count = 0

    def load_model(self):
        """
        加载模型和tokenizer
        实现容错机制避免网络或内存问题，增强Windows兼容性
        """
        logger.info("开始加载DeepSeek模型")

        # Windows特殊处理
        if platform.system() == "Windows":
            # 设置环境变量
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免OpenMP冲突

            # 增加超时时间
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(300)  # 5分钟超时

        model_path = str(self.config.LOCAL_MODEL_PATH)

        for attempt in range(self.config.MAX_RETRIES):
            try:
                # 检查可用内存
                import psutil
                available_memory = psutil.virtual_memory().available / 1024 ** 3
                if available_memory < 8:  # 少于8GB
                    logger.warning(f"可用内存不足: {available_memory:.1f}GB")

                # 加载tokenizer
                logger.info(f"加载tokenizer (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.tokenizer.padding_side = 'left'
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # 加载模型
                logger.info(f"加载模型权重 (尝试 {attempt + 1}/{self.config.MAX_RETRIES})")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=getattr(torch, self.config.TORCH_DTYPE),
                    device_map=self.device,
                    attn_implementation=self.config.ATTENTION_IMPL,
                    trust_remote_code=True  # DeepSeek可能需要
                )

                # 设置生成配置
                self.model.generation_config = GenerationConfig.from_pretrained(model_path)
                self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

                # 设置为评估模式
                torch.set_grad_enabled(False)
                self.model.eval()

                # 验证模型加载
                self._validate_model()

                # Windows: 恢复默认超时
                if platform.system() == "Windows":
                    socket.setdefaulttimeout(original_timeout)

                logger.info("模型加载成功")
                return True

            except Exception as e:
                # Windows特殊错误处理
                if platform.system() == "Windows" and "timeout" in str(e).lower():
                    logger.warning("网络超时，可能是防火墙或网络问题")

                logger.warning(f"模型加载失败 (尝试 {attempt + 1}/{self.config.MAX_RETRIES}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    self.clear_memory()
                else:
                    logger.error("模型加载彻底失败")
                    raise RuntimeError(f"模型加载失败: {e}")

        return False

    def _validate_model(self):
        """验证模型加载正确性"""
        try:
            # 简单推理测试
            test_input = "Hello"
            inputs = self.tokenizer(test_input, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            logger.info("模型验证通过")

        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            raise RuntimeError("模型验证失败")

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = None) -> Tuple[str, List[int], List[float]]:
        """生成文本并返回logprobs - 简化日志版本"""
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_NEW_TOKENS

        # 定期内存清理
        self._generation_count += 1
        if self._generation_count % 10 == 0:
            self.clear_memory()

        # 检查内存使用
        self._check_memory_usage()

        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成配置
            generation_config = {
                "do_sample": self.config.DO_SAMPLE,
                "num_beams": self.config.NUM_BEAMS,
                "temperature": self.config.TEMPERATURE if self.config.TEMPERATURE > 0 else None,
                "max_new_tokens": max_new_tokens,
                "output_hidden_states": True,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            if self.config.TEMPERATURE <= 0:
                generation_config.pop("temperature", None)

            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            # 提取生成的token IDs
            generated_token_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]

            # 提取logits并计算概率
            logits = outputs.scores
            probs = []

            for i, logit in enumerate(logits):
                top_logits, _ = torch.sort(logit[0], descending=True)
                top_logits = top_logits[:5]
                top_probs = torch.nn.functional.softmax(top_logits, dim=-1)
                probs.append(top_probs[0].item())

            # 解码生成的文本
            generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            return generated_text, generated_token_ids.tolist(), probs

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU内存不足，清理内存后重试")
            self.clear_memory()
            if max_new_tokens > 256:
                return self.generate_with_logprobs(prompt, max_new_tokens // 2)
            else:
                raise RuntimeError("GPU内存不足且无法进一步降低生成长度")
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise RuntimeError(f"文本生成失败: {e}")

    def _check_memory_usage(self):
        """检查内存使用情况"""
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_allocated > self.config.MEMORY_THRESHOLD:
                    logger.warning(f"GPU内存使用率过高: {memory_allocated:.2%}")
                    self.clear_memory()
            except:
                pass  # 忽略GPU内存检查错误

    def clear_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("内存清理完成")

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        import psutil

        usage = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            try:
                usage["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024 ** 3
                usage["gpu_memory_cached_gb"] = torch.cuda.memory_reserved() / 1024 ** 3
                usage["gpu_memory_percent"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass  # 忽略GPU内存获取错误

        return usage

    def health_check(self) -> bool:
        """模型健康检查"""
        try:
            if self.model is None or self.tokenizer is None:
                return False

            # 简单推理测试
            test_text, _, _ = self.generate_with_logprobs("Test", max_new_tokens=5)
            return len(test_text) > 0

        except Exception as e:
            logger.warning(f"模型健康检查失败: {e}")
            return False