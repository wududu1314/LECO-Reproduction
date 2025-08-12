"""
置信度计算模块
严格实现论文3.1节中的三因子置信度计算方法
"""

import numpy as np
import scipy.stats
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    def __init__(self, config):
        self.config = config

    def calculate_step_confidence(self, token_probs: List[float], step_text: str) -> Tuple[List[float], float]:
        """
        计算单个步骤的置信度
        严格按照论文公式4实现: siscore = avg_scorei + trans_scorei - diver_scorei

        Args:
            token_probs: token概率列表
            step_text: 步骤文本

        Returns:
            ([avg_score, trans_score, -diver_score], total_confidence)
        """
        if len(token_probs) == 0:
            logger.warning("空的token概率列表")
            return [0.0, 0.0, 0.0], 0.0

        try:
            # 1. 平均token置信度 (公式1)
            avg_score = np.mean(token_probs)

            # 2. 步骤间转换分数 (公式3)
            K = self.config.CONFIDENCE_K  # 论文中K=3
            if step_text.strip().startswith("Step"):
                # 对于Step开头的句子，跳过"Step"相关token - 论文3.1节描述
                if len(token_probs) > 3 + K:
                    trans_score = np.mean(token_probs[3:3 + K])
                else:
                    trans_score = np.mean(token_probs[3:]) if len(token_probs) > 3 else avg_score
            else:
                trans_score = np.mean(token_probs[:K]) if len(token_probs) >= K else avg_score

            # 3. 发散分数（KL散度） (公式2)
            diver_score = self._calculate_kl_divergence(token_probs)

            # 总置信度：论文公式4
            total_confidence = avg_score + trans_score - diver_score

            return [avg_score, trans_score, -diver_score], total_confidence

        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return [0.0, 0.0, 0.0], 0.0

    def _calculate_kl_divergence(self, token_probs: List[float]) -> float:
        """
        计算KL散度 - 严格按照论文公式2实现
        diver_scorei = ln(KLDτ(Pi, U) + 1)
        """
        step_len = len(token_probs)
        if step_len <= 1:
            return 0.0

        try:
            # 创建均匀分布U - 论文公式2
            uniform_dist = np.array([1.0 / step_len] * step_len)

            # 标准化token概率Pi
            token_probs_array = np.array(token_probs)
            # 避免零概率导致的数值问题
            token_probs_array = np.maximum(token_probs_array, 1e-10)
            token_probs_normalized = token_probs_array / np.sum(token_probs_array)

            # 计算KL散度
            kl_div = scipy.stats.entropy(token_probs_normalized, uniform_dist)

            # 应用论文中的变换：ln(KL^(1/τ) + 1)
            tau = self.config.DIVERGENCE_TAU  # τ = 0.3
            result = np.log(np.power(kl_div, 1 / tau) + 1)

            # 处理数值异常
            if np.isnan(result) or np.isinf(result):
                logger.warning("KL散度计算出现数值异常")
                return 0.0

            return float(result)

        except Exception as e:
            logger.warning(f"KL散度计算异常: {e}")
            return 0.0

    def extract_step_info(self, token_ids: List[int], token_probs: List[float],
                          generated_text: str, tokenizer) -> List[Dict]:
        """
        从生成的文本中提取步骤信息和置信度
        实现与原论文extract_step_info完全一致的步骤分割逻辑
        """
        step_info = []
        current_step_tokens = []
        current_step_probs = []
        current_step_text = ""

        try:
            for i, (token_id, prob) in enumerate(zip(token_ids, token_probs)):
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                current_step_text += token_text
                current_step_tokens.append(token_id)
                current_step_probs.append(prob)

                # 检查是否为步骤结束（换行符token_id通常是185或10）- 与原论文一致
                if token_id in [185, 10, 13] or i == len(token_ids) - 1:
                    if len(current_step_probs) > 0 and len(current_step_text.strip()) > 0:
                        confidence_components, total_confidence = self.calculate_step_confidence(
                            current_step_probs, current_step_text
                        )

                        step_info.append({
                            "id": len(step_info),
                            "sentence": current_step_text.strip(),
                            "confidence": confidence_components,
                            "total_confidence": total_confidence,
                            "token_count": len(current_step_tokens),
                            "avg_token_prob": np.mean(current_step_probs)
                        })

                    # 重置当前步骤
                    current_step_tokens = []
                    current_step_probs = []
                    current_step_text = ""

        except Exception as e:
            logger.error(f"步骤信息提取失败: {e}")
            # 返回至少一个空步骤以避免后续处理失败
            if not step_info:
                step_info = [{
                    "id": 0,
                    "sentence": generated_text,
                    "confidence": [0.0, 0.0, 0.0],
                    "total_confidence": 0.0,
                    "token_count": len(token_ids),
                    "avg_token_prob": np.mean(token_probs) if token_probs else 0.0
                }]

        return step_info

    def validate_confidence_calculation(self, step_info: List[Dict]) -> bool:
        """验证置信度计算的合理性"""
        try:
            for step in step_info:
                conf = step["confidence"]
                total = step["total_confidence"]

                # 检查置信度组件是否为数值
                if not all(isinstance(c, (int, float)) and not np.isnan(c) for c in conf):
                    logger.warning(f"步骤{step['id']}置信度包含无效值")
                    return False

                # 检查总置信度计算是否正确
                expected_total = conf[0] + conf[1] + conf[2]  # avg + trans - diver (注意diver已是负值)
                if abs(total - expected_total) > 1e-6:
                    logger.warning(f"步骤{step['id']}总置信度计算不一致")
                    return False

            return True

        except Exception as e:
            logger.error(f"置信度验证失败: {e}")
            return False