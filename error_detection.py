"""
错误检测策略模块
包含原版LECO和简化版改进错误检测方法

改进策略：使用统计学阈值 threshold = mean - 1.5 * std
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class ErrorDetector:
    def __init__(self, config):
        self.config = config
        self.detection_stats = {
            "original_detections": 0,
            "improved_detections": 0,
            "threshold_activations": 0,
            "fallback_activations": 0
        }

    def find_error_step_original(self, step_info: List[Dict]) -> Tuple[int, List[Dict]]:
        """
        原版LECO错误检测方法
        严格按照原论文find_error_step函数实现
        返回最低置信度步骤的索引
        """
        try:
            # 过滤掉系统步骤 - 与原论文一致
            valid_steps = []
            valid_indices = []

            for i, step in enumerate(step_info):
                sentence = step["sentence"].lower()
                # 过滤条件与原论文完全一致
                if ("step by step" not in sentence and
                        "the answer is" not in sentence and
                        len(sentence.strip()) > 0):
                    valid_steps.append(step)
                    valid_indices.append(i)

            if not valid_steps:
                logger.warning("没有有效步骤用于错误检测")
                return 0, []

            # 按总置信度排序 - 原论文逻辑
            sorted_steps = sorted(
                zip(valid_steps, valid_indices),
                key=lambda x: x[0]["total_confidence"]
            )

            # 返回置信度最低的步骤
            error_step_idx = sorted_steps[0][1]
            potential_error_steps = [step[0] for step in sorted_steps]

            self.detection_stats["original_detections"] += 1
            logger.debug(f"原版方法检测到错误步骤: {error_step_idx}")
            return error_step_idx, potential_error_steps

        except Exception as e:
            logger.error(f"原版错误检测失败: {e}")
            return 0, []

    def find_error_step_improved(self, step_info: List[Dict]) -> Tuple[int, List[Dict]]:
        """
        改进的错误检测策略

        核心公式：threshold = mean(confidences) - 1.5 * std(confidences)
        策略：找到第一个低于阈值的步骤
        """
        try:
            # 1. 过滤掉系统步骤
            valid_steps = []
            valid_indices = []

            for i, step in enumerate(step_info):
                sentence = step["sentence"].lower()
                if ("step by step" not in sentence and
                        "the answer is" not in sentence and
                        len(sentence.strip()) > 0):
                    valid_steps.append(step)
                    valid_indices.append(i)

            if len(valid_steps) < 3:
                # 步骤太少，使用原版方法
                logger.debug("步骤数量不足，回退到原版方法")
                self.detection_stats["fallback_activations"] += 1
                return self.find_error_step_original(step_info)

            # 2. 计算所有步骤的置信度
            confidences = np.array([step["total_confidence"] for step in valid_steps])

            # 3. 使用用户指定的统计学阈值
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            threshold = mean_conf - 1.5 * std_conf
            threshold = max(0.05, threshold)  # 下界保护

            logger.debug(f"改进方法 - mean: {mean_conf:.4f}, std: {std_conf:.4f}, threshold: {threshold:.4f}")

            # 4. 优先处理严重错误（第一个低于阈值的步骤）
            for i, confidence in enumerate(confidences):
                if confidence < threshold:
                    error_step_idx = valid_indices[i]
                    logger.debug(f"检测到严重错误步骤: {error_step_idx} (置信度: {confidence:.4f})")
                    self.detection_stats["threshold_activations"] += 1

                    # 返回按置信度排序的潜在错误步骤
                    potential_errors = sorted(valid_steps, key=lambda x: x["total_confidence"])
                    return error_step_idx, potential_errors

            # 5. 如果没有严重错误，处理置信度最低的步骤
            min_conf_idx = np.argmin(confidences)
            error_step_idx = valid_indices[min_conf_idx]
            logger.debug(f"未检测到严重错误，选择最低置信度步骤: {error_step_idx}")

            potential_errors = sorted(valid_steps, key=lambda x: x["total_confidence"])
            self.detection_stats["improved_detections"] += 1
            return error_step_idx, potential_errors

        except Exception as e:
            logger.error(f"改进错误检测失败: {e}")
            # 容错：返回原版方法的结果
            self.detection_stats["fallback_activations"] += 1
            return self.find_error_step_original(step_info)

    def get_detection_statistics(self) -> Dict:
        """获取检测统计信息"""
        total = self.detection_stats["original_detections"] + self.detection_stats["improved_detections"]
        if total > 0:
            return {
                **self.detection_stats,
                "threshold_activation_rate": self.detection_stats["threshold_activations"] / total,
                "fallback_rate": self.detection_stats["fallback_activations"] / total
            }
        return self.detection_stats

    def should_continue_iteration(self, prev_answer: str, current_answer: str,
                                  iteration_num: int) -> bool:
        """
        判断是否应该继续迭代
        与原论文Algorithm 1的停止条件完全一致
        """
        try:
            # 如果答案相同或达到最大迭代次数，停止
            if prev_answer == current_answer:
                logger.debug(f"答案收敛，停止迭代 (轮次: {iteration_num})")
                return False

            if iteration_num >= self.config.MAX_ITERATIONS:
                logger.debug(f"达到最大迭代次数，停止迭代 (轮次: {iteration_num})")
                return False

            return True

        except Exception as e:
            logger.error(f"迭代条件判断失败: {e}")
            return False

    def majority_vote(self, candidates: List) -> any:
        """
        多数投票选择最终答案
        与原论文majorElement函数完全一致：给最新答案额外权重
        """
        if not candidates:
            logger.warning("空的候选答案列表")
            return None

        try:
            vote_dict = {}
            candidates_reversed = list(reversed(candidates))

            for i, candidate in enumerate(candidates_reversed):
                # 最新答案权重为1.5，其他为1.0 - 与原论文完全一致
                weight = 1.5 if i == 0 else 1.0

                # 处理NaN值
                if isinstance(candidate, float) and np.isnan(candidate):
                    continue

                vote_dict[candidate] = vote_dict.get(candidate, 0) + weight

            # 返回得票最高的答案
            if vote_dict:
                winner = max(vote_dict.items(), key=lambda x: x[1])
                logger.debug(f"多数投票结果: {winner[0]} (得票: {winner[1]})")
                return winner[0]
            else:
                # 如果所有候选都无效，返回最后一个非NaN答案
                for candidate in reversed(candidates):
                    if not (isinstance(candidate, float) and np.isnan(candidate)):
                        return candidate
                return candidates[-1] if candidates else None

        except Exception as e:
            logger.error(f"多数投票失败: {e}")
            return candidates[-1] if candidates else None

    def analyze_error_pattern(self, original_answer: any, improved_answer: any,
                              ground_truth: any) -> str:
        """
        分析错误模式变化
        用于实验结果统计
        """

        def is_correct(pred, truth):
            try:
                if isinstance(truth, (int, float)):
                    if isinstance(pred, (int, float)) and not np.isnan(pred):
                        return abs(pred - truth) < 1e-6
                    return False
                else:
                    return str(pred).strip() == str(truth).strip()
            except:
                return False

        orig_correct = is_correct(original_answer, ground_truth)
        impr_correct = is_correct(improved_answer, ground_truth)

        if not orig_correct and impr_correct:
            return "W2R"  # Wrong to Right
        elif orig_correct and not impr_correct:
            return "R2W"  # Right to Wrong
        elif not orig_correct and not impr_correct:
            return "W2W"  # Wrong to Wrong
        else:
            return "R2R"  # Right to Right

    def compare_detection_methods(self, step_info: List[Dict]) -> Dict:
        """
        比较两种错误检测方法的结果
        用于分析改进效果
        """
        try:
            orig_error_idx, orig_potential = self.find_error_step_original(step_info)
            impr_error_idx, impr_potential = self.find_error_step_improved(step_info)

            return {
                "original_method": {
                    "error_step_idx": orig_error_idx,
                    "error_confidence": orig_potential[0]["total_confidence"] if orig_potential else 0.0
                },
                "improved_method": {
                    "error_step_idx": impr_error_idx,
                    "error_confidence": impr_potential[0]["total_confidence"] if impr_potential else 0.0
                },
                "methods_agree": orig_error_idx == impr_error_idx
            }

        except Exception as e:
            logger.error(f"错误检测方法比较失败: {e}")
            return {}