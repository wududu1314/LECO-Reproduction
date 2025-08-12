"""
评估和结果分析模块
"""

import json
import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_accuracy(self, results: List[Dict]) -> Dict[str, float]:
        """
        计算准确率
        严格按照原论文的评估标准
        """
        total = len(results)
        correct = 0
        invalid = 0

        for result in results:
            try:
                if self._is_answer_correct(result["prediction"], result["ground_truth"]):
                    correct += 1
                elif self._is_invalid_answer(result["prediction"]):
                    invalid += 1
            except Exception as e:
                logger.warning(f"答案评估异常: {e}")
                invalid += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "invalid": invalid,
            "valid_rate": (total - invalid) / total if total > 0 else 0.0
        }

    def _is_answer_correct(self, pred, truth) -> bool:
        """
        判断答案是否正确
        与原论文评估标准保持一致
        """
        try:
            # 对于数值类型（GSM8K）
            if isinstance(truth, (int, float)):
                if isinstance(pred, (int, float)):
                    if np.isnan(pred) or np.isnan(truth):
                        return False
                    return abs(pred - truth) < 1e-6
                else:
                    return False
            # 对于字符串类型（MATH）
            else:
                pred_str = str(pred).strip()
                truth_str = str(truth).strip()
                return pred_str == truth_str
        except Exception as e:
            logger.warning(f"答案比较异常: {e}")
            return False

    def _is_invalid_answer(self, pred) -> bool:
        """判断答案是否无效"""
        if pred is None:
            return True
        if isinstance(pred, float) and np.isnan(pred):
            return True
        if isinstance(pred, str) and pred.strip() == "":
            return True
        return False

    def calculate_token_consumption(self, results: List[Dict]) -> Dict[str, int]:
        """
        计算token消耗
        与原论文统计方法一致
        """
        total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
        total_iterations = sum(r.get("iterations", 1) for r in results)

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_iterations": total_iterations / len(results) if results else 0,
            "avg_tokens_per_question": (total_input_tokens + total_output_tokens) / len(results) if results else 0
        }

    def analyze_error_patterns(self, original_results: List[Dict],
                               improved_results: List[Dict]) -> Dict[str, int]:
        """
        分析错误模式变化
        用于评估改进效果
        """
        patterns = {
            "W2R": 0,  # Wrong to Right
            "R2W": 0,  # Right to Wrong
            "W2W": 0,  # Wrong to Wrong
            "R2R": 0  # Right to Right
        }

        if len(original_results) != len(improved_results):
            logger.error("原版和改进版结果数量不匹配")
            return patterns

        for orig, impr in zip(original_results, improved_results):
            try:
                orig_correct = self._is_answer_correct(orig["prediction"], orig["ground_truth"])
                impr_correct = self._is_answer_correct(impr["prediction"], impr["ground_truth"])

                if not orig_correct and impr_correct:
                    patterns["W2R"] += 1
                elif orig_correct and not impr_correct:
                    patterns["R2W"] += 1
                elif not orig_correct and not impr_correct:
                    patterns["W2W"] += 1
                else:
                    patterns["R2R"] += 1

            except Exception as e:
                logger.warning(f"错误模式分析异常: {e}")
                continue

        return patterns

    def calculate_statistical_significance(self, original_results: List[Dict],
                                           improved_results: List[Dict]) -> Dict:
        """
        计算统计显著性
        使用配对t检验评估改进是否显著
        """
        try:
            from scipy import stats

            # 提取准确率数据
            orig_correct = [1 if self._is_answer_correct(r["prediction"], r["ground_truth"]) else 0
                            for r in original_results]
            impr_correct = [1 if self._is_answer_correct(r["prediction"], r["ground_truth"]) else 0
                            for r in improved_results]

            # 配对t检验
            statistic, p_value = stats.ttest_rel(impr_correct, orig_correct)

            # 效应大小（Cohen's d）
            diff = np.array(impr_correct) - np.array(orig_correct)
            effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

            return {
                "t_statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant": p_value < 0.05,
                "improvement_significant": p_value < 0.05 and statistic > 0
            }

        except ImportError:
            logger.warning("scipy未安装，跳过统计显著性检验")
            return {}
        except Exception as e:
            logger.error(f"统计显著性计算失败: {e}")
            return {}

    def _validate_improvement(self, orig_acc: float, impr_acc: float, sample_size: int) -> Dict:
        """验证改进的统计显著性"""
        improvement = impr_acc - orig_acc

        # 简单的显著性检验
        if sample_size >= 100:
            # 大样本近似
            se = np.sqrt((orig_acc * (1 - orig_acc) + impr_acc * (1 - impr_acc)) / sample_size)
            z_score = improvement / se if se > 0 else 0
            significant = abs(z_score) > 1.96  # 95%置信水平
        else:
            significant = improvement > 0.01  # 1%改进阈值
            z_score = 0

        return {
            "improvement_percent": improvement * 100,
            "statistically_significant": significant,
            "z_score": z_score,
            "practical_significance": improvement > 0.005  # 0.5%实际意义阈值
        }

    def _compare_with_paper_expectation(self, our_baseline: float, our_improved: float, dataset: str) -> Dict:
        """与论文期望结果对比"""
        from config import Config

        if dataset in ["gsm8k"]:
            paper_baseline = Config.PAPER_BASELINES["deepseek-math-7b-rl"]["gsm8k_complex"]
            paper_improved = Config.PAPER_BASELINES["deepseek-math-7b-rl"]["gsm8k_leco"]
        elif dataset in ["math"]:
            paper_baseline = Config.PAPER_BASELINES["deepseek-math-7b-rl"]["math_complex"]
            paper_improved = Config.PAPER_BASELINES["deepseek-math-7b-rl"]["math_leco"]
        else:
            return {"error": "未知数据集"}

        paper_improvement = paper_improved - paper_baseline
        our_improvement = our_improved - our_baseline

        return {
            "paper_baseline": paper_baseline,
            "our_baseline": our_baseline,
            "baseline_diff": our_baseline - paper_baseline,
            "paper_improvement": paper_improvement,
            "our_improvement": our_improvement,
            "improvement_ratio": our_improvement / paper_improvement if paper_improvement > 0 else 0,
            "meets_expectation": our_improvement >= paper_improvement * 0.5  # 达到论文50%的改进
        }

    def generate_comparison_report(self, original_results: List[Dict],
                                   improved_results: List[Dict],
                                   dataset_name: str) -> Dict:
        """
        生成完整的对比报告
        专注于方法间比较，不再对比论文基准
        """
        try:
            # 准确率对比
            orig_acc = self.evaluate_accuracy(original_results)
            impr_acc = self.evaluate_accuracy(improved_results)

            # Token消耗对比
            orig_tokens = self.calculate_token_consumption(original_results)
            impr_tokens = self.calculate_token_consumption(improved_results)

            # 错误模式分析
            error_patterns = self.analyze_error_patterns(original_results, improved_results)

            # 统计显著性分析
            significance = self.calculate_statistical_significance(original_results, improved_results)

            # 改进效果显著性验证
            improvement_validation = self._validate_improvement(
                orig_acc["accuracy"],
                impr_acc["accuracy"],
                len(original_results)
            )

            # 确定测试类型
            sample_size = len(original_results)
            if sample_size == 700:
                test_type = "custom_700"
            elif sample_size <= 10:
                test_type = "quick"
            else:
                test_type = "other"

            report = {
                "dataset": dataset_name,
                "experiment_info": {
                    "sample_size": sample_size,
                    "test_type": test_type,
                    "random_seed": self.config.RANDOM_SEED,
                    "model_name": self.config.MODEL_NAME
                },
                "original_method": {
                    "accuracy": orig_acc["accuracy"],
                    "correct": orig_acc["correct"],
                    "total": orig_acc["total"],
                    "invalid": orig_acc["invalid"],
                    "token_consumption": orig_tokens
                },
                "improved_method": {
                    "accuracy": impr_acc["accuracy"],
                    "correct": impr_acc["correct"],
                    "total": impr_acc["total"],
                    "invalid": impr_acc["invalid"],
                    "token_consumption": impr_tokens
                },
                "improvement": {
                    "accuracy_gain": impr_acc["accuracy"] - orig_acc["accuracy"],
                    "accuracy_gain_percent": (
                            (impr_acc["accuracy"] - orig_acc["accuracy"]) / orig_acc["accuracy"] * 100
                    ) if orig_acc["accuracy"] > 0 else 0,
                    "token_reduction": orig_tokens["total_tokens"] - impr_tokens["total_tokens"],
                    "token_reduction_percent": (
                            (orig_tokens["total_tokens"] - impr_tokens["total_tokens"]) / orig_tokens[
                        "total_tokens"] * 100
                    ) if orig_tokens["total_tokens"] > 0 else 0,
                    "iteration_reduction": orig_tokens["avg_iterations"] - impr_tokens["avg_iterations"]
                },
                "error_patterns": error_patterns,
                "statistical_analysis": significance,
                "improvement_validation": improvement_validation
            }

            return report

        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            raise RuntimeError("报告生成失败")

    def generate_combined_summary(self, all_results: Dict, test_mode: str) -> Dict:
        """生成两个数据集的汇总报告"""
        try:
            summary = {
                "experiment_info": {
                    "test_mode": test_mode,
                    "datasets": list(all_results.keys()),
                    "total_samples": sum(len(all_results[ds]["original"]) for ds in all_results),
                    "random_seed": self.config.RANDOM_SEED
                },
                "dataset_results": {},
                "overall_improvement": {}
            }

            # 汇总各数据集结果
            total_orig_correct = 0
            total_orig_total = 0
            total_impr_correct = 0
            total_impr_total = 0
            total_orig_tokens = 0
            total_impr_tokens = 0

            for dataset, results in all_results.items():
                comparison = results["comparison"]
                summary["dataset_results"][dataset] = {
                    "original_accuracy": comparison["original_method"]["accuracy"],
                    "improved_accuracy": comparison["improved_method"]["accuracy"],
                    "improvement": comparison["improvement"]["accuracy_gain"],
                    "sample_size": comparison["experiment_info"]["sample_size"]
                }

                # 累加总体统计
                total_orig_correct += comparison["original_method"]["correct"]
                total_orig_total += comparison["original_method"]["total"]
                total_impr_correct += comparison["improved_method"]["correct"]
                total_impr_total += comparison["improved_method"]["total"]
                total_orig_tokens += comparison["original_method"]["token_consumption"]["total_tokens"]
                total_impr_tokens += comparison["improved_method"]["token_consumption"]["total_tokens"]

            # 计算整体改进
            overall_orig_acc = total_orig_correct / total_orig_total if total_orig_total > 0 else 0
            overall_impr_acc = total_impr_correct / total_impr_total if total_impr_total > 0 else 0

            summary["overall_improvement"] = {
                "original_accuracy": overall_orig_acc,
                "improved_accuracy": overall_impr_acc,
                "accuracy_gain": overall_impr_acc - overall_orig_acc,
                "token_reduction": total_orig_tokens - total_impr_tokens,
                "total_samples": total_orig_total
            }

            return summary

        except Exception as e:
            logger.error(f"汇总报告生成失败: {e}")
            raise

    def save_report(self, report: Dict, output_path):
        """保存报告"""
        try:
            from pathlib import Path
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)

            # 保存JSON格式 - 修复布尔值序列化问题
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)  # 添加default=str

            # 保存可读文本格式
            txt_path = output_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(self._format_report_text(report))

            logger.info(f"报告保存完成: {json_path}, {txt_path}")

        except Exception as e:
            logger.error(f"报告保存失败: {e}")
            raise RuntimeError("报告保存失败")

    def _format_report_text(self, report: Dict) -> str:
        """格式化报告文本"""
        text = f"""
# LECO方法对比实验报告

## 实验信息
- 数据集: {report['dataset']}
- 样本数量: {report['experiment_info']['sample_size']}
- 随机种子: {report['experiment_info']['random_seed']}
- 模型: {report['experiment_info']['model_name']}

## 准确率对比
- 原版LECO: {report['original_method']['accuracy']:.4f} ({report['original_method']['correct']}/{report['original_method']['total']})
- 改进版本: {report['improved_method']['accuracy']:.4f} ({report['improved_method']['correct']}/{report['improved_method']['total']})
- 准确率提升: {report['improvement']['accuracy_gain']:.4f} ({report['improvement']['accuracy_gain_percent']:.2f}%)

## Token消耗对比
- 原版LECO: {report['original_method']['token_consumption']['total_tokens']:,} tokens
- 改进版本: {report['improved_method']['token_consumption']['total_tokens']:,} tokens  
- Token节省: {report['improvement']['token_reduction']:,} tokens ({report['improvement']['token_reduction_percent']:.2f}%)

## 迭代次数对比
- 原版LECO: {report['original_method']['token_consumption']['avg_iterations']:.2f} 轮
- 改进版本: {report['improved_method']['token_consumption']['avg_iterations']:.2f} 轮
- 迭代减少: {report['improvement']['iteration_reduction']:.2f} 轮

## 错误模式分析
- 错误→正确 (W2R): {report['error_patterns']['W2R']}
- 正确→错误 (R2W): {report['error_patterns']['R2W']} 
- 错误→错误 (W2W): {report['error_patterns']['W2W']}
- 正确→正确 (R2R): {report['error_patterns']['R2R']}

## 统计显著性分析
"""

        if report.get('statistical_analysis'):
            stats = report['statistical_analysis']
            text += f"""- t统计量: {stats.get('t_statistic', 'N/A'):.4f}
- p值: {stats.get('p_value', 'N/A'):.4f}
- 效应大小: {stats.get('effect_size', 'N/A'):.4f}
- 改进显著: {'是' if stats.get('improvement_significant', False) else '否'}
"""
        else:
            text += "- 统计分析未完成\n"

        if report.get('paper_comparison'):
            paper = report['paper_comparison']
            text += f"""
## 与论文基准对比
- 论文基准准确率: {paper.get('paper_baseline', 'N/A'):.2f}%
- 我们的基准准确率: {paper.get('our_baseline', 'N/A'):.2f}%
- 基准差异: {paper.get('baseline_diff', 'N/A'):.2f}%
- 论文改进幅度: {paper.get('paper_improvement', 'N/A'):.2f}%
- 我们的改进幅度: {paper.get('our_improvement', 'N/A'):.2f}%
- 达到预期: {'是' if paper.get('meets_expectation', False) else '否'}
"""

        return text

    def create_visualization(self, report: Dict, output_dir: Path):
        """创建可视化图表"""
        try:
            output_dir.mkdir(exist_ok=True, parents=True)

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 子图1：准确率对比
            methods = ['原版LECO', '改进版本']
            accuracies = [
                report['original_method']['accuracy'],
                report['improved_method']['accuracy']
            ]
            bars1 = ax1.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
            ax1.set_title('准确率对比')
            ax1.set_ylabel('准确率')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')

            # 子图2：Token消耗对比
            token_counts = [
                report['original_method']['token_consumption']['total_tokens'],
                report['improved_method']['token_consumption']['total_tokens']
            ]
            bars2 = ax2.bar(methods, token_counts, color=['skyblue', 'lightcoral'])
            ax2.set_title('Token消耗对比')
            ax2.set_ylabel('Token数量')
            for i, v in enumerate(token_counts):
                ax2.text(i, v + max(token_counts) * 0.01, f'{v:,}', ha='center')

            # 子图3：错误模式分析
            patterns = report['error_patterns']
            pattern_labels = ['W2R', 'R2W', 'W2W', 'R2R']
            pattern_values = [patterns[label] for label in pattern_labels]
            colors = ['green', 'red', 'orange', 'blue']
            ax3.pie(pattern_values, labels=pattern_labels, colors=colors, autopct='%1.1f%%')
            ax3.set_title('错误模式分布')

            # 子图4：改进效果总结
            improvement_metrics = ['准确率提升', 'Token节省率', '迭代减少']
            improvement_values = [
                report['improvement']['accuracy_gain_percent'],
                report['improvement']['token_reduction_percent'],
                abs(report['improvement']['iteration_reduction']) * 100 /
                report['original_method']['token_consumption']['avg_iterations'] if
                report['original_method']['token_consumption']['avg_iterations'] > 0 else 0
            ]
            bars4 = ax4.bar(improvement_metrics, improvement_values, color='green', alpha=0.7)
            ax4.set_title('改进效果 (%)')
            ax4.set_ylabel('改进百分比')
            for i, v in enumerate(improvement_values):
                ax4.text(i, v + max(improvement_values) * 0.01, f'{v:.1f}%', ha='center')

            plt.tight_layout()
            plt.savefig(output_dir / f"{report['dataset']}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"可视化图表已保存到: {output_dir}")

        except Exception as e:
            logger.warning(f"可视化创建失败: {e}")

    def create_detailed_excel(self, original_results: List[Dict], improved_results: List[Dict],
                              output_path):
        """创建详细的Excel报告"""
        try:
            import pandas as pd

            # 分离GSM8K和MATH结果
            gsm8k_orig = [r for r in original_results if r["dataset"] == "gsm8k"]
            gsm8k_impr = [r for r in improved_results if r["dataset"] == "gsm8k"]
            math_orig = [r for r in original_results if r["dataset"] == "math"]
            math_impr = [r for r in improved_results if r["dataset"] == "math"]

            with pd.ExcelWriter(f"{output_path}.xlsx", engine='xlsxwriter') as writer:

                # 总体汇总sheet
                summary_data = []

                # GSM8K汇总
                if gsm8k_orig and gsm8k_impr:
                    gsm8k_orig_acc = self.evaluate_accuracy(gsm8k_orig)
                    gsm8k_impr_acc = self.evaluate_accuracy(gsm8k_impr)
                    gsm8k_orig_tokens = self.calculate_token_consumption(gsm8k_orig)
                    gsm8k_impr_tokens = self.calculate_token_consumption(gsm8k_impr)

                    summary_data.append({
                        "数据集": "GSM8K",
                        "样本数": len(gsm8k_orig),
                        "原版准确率": f"{gsm8k_orig_acc['accuracy']:.4f}",
                        "原版正确数": gsm8k_orig_acc['correct'],
                        "改进准确率": f"{gsm8k_impr_acc['accuracy']:.4f}",
                        "改进正确数": gsm8k_impr_acc['correct'],
                        "准确率提升": f"{gsm8k_impr_acc['accuracy'] - gsm8k_orig_acc['accuracy']:+.4f}",
                        "原版Token": gsm8k_orig_tokens['total_tokens'],
                        "改进Token": gsm8k_impr_tokens['total_tokens'],
                        "Token变化": gsm8k_orig_tokens['total_tokens'] - gsm8k_impr_tokens['total_tokens'],
                        "原版平均迭代": f"{gsm8k_orig_tokens['avg_iterations']:.2f}",
                        "改进平均迭代": f"{gsm8k_impr_tokens['avg_iterations']:.2f}"
                    })

                # MATH汇总
                if math_orig and math_impr:
                    math_orig_acc = self.evaluate_accuracy(math_orig)
                    math_impr_acc = self.evaluate_accuracy(math_impr)
                    math_orig_tokens = self.calculate_token_consumption(math_orig)
                    math_impr_tokens = self.calculate_token_consumption(math_impr)

                    summary_data.append({
                        "数据集": "MATH",
                        "样本数": len(math_orig),
                        "原版准确率": f"{math_orig_acc['accuracy']:.4f}",
                        "原版正确数": math_orig_acc['correct'],
                        "改进准确率": f"{math_impr_acc['accuracy']:.4f}",
                        "改进正确数": math_impr_acc['correct'],
                        "准确率提升": f"{math_impr_acc['accuracy'] - math_orig_acc['accuracy']:+.4f}",
                        "原版Token": math_orig_tokens['total_tokens'],
                        "改进Token": math_impr_tokens['total_tokens'],
                        "Token变化": math_orig_tokens['total_tokens'] - math_impr_tokens['total_tokens'],
                        "原版平均迭代": f"{math_orig_tokens['avg_iterations']:.2f}",
                        "改进平均迭代": f"{math_impr_tokens['avg_iterations']:.2f}"
                    })

                # 整体汇总
                if len(summary_data) > 0:
                    total_orig_acc = self.evaluate_accuracy(original_results)
                    total_impr_acc = self.evaluate_accuracy(improved_results)
                    total_orig_tokens = self.calculate_token_consumption(original_results)
                    total_impr_tokens = self.calculate_token_consumption(improved_results)

                    summary_data.append({
                        "数据集": "总计",
                        "样本数": len(original_results),
                        "原版准确率": f"{total_orig_acc['accuracy']:.4f}",
                        "原版正确数": total_orig_acc['correct'],
                        "改进准确率": f"{total_impr_acc['accuracy']:.4f}",
                        "改进正确数": total_impr_acc['correct'],
                        "准确率提升": f"{total_impr_acc['accuracy'] - total_orig_acc['accuracy']:+.4f}",
                        "原版Token": total_orig_tokens['total_tokens'],
                        "改进Token": total_impr_tokens['total_tokens'],
                        "Token变化": total_orig_tokens['total_tokens'] - total_impr_tokens['total_tokens'],
                        "原版平均迭代": f"{total_orig_tokens['avg_iterations']:.2f}",
                        "改进平均迭代": f"{total_impr_tokens['avg_iterations']:.2f}"
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="汇总", index=False)

                # 详细结果sheet
                detailed_data = []
                for i, (orig, impr) in enumerate(zip(original_results, improved_results)):
                    detailed_data.append({
                        "序号": i + 1,
                        "数据集": orig["dataset"],
                        "问题": orig["question"][:100] + "..." if len(orig["question"]) > 100 else orig["question"],
                        "标准答案": str(orig["ground_truth"]),
                        "原版预测": str(orig["prediction"]),
                        "原版正确": self._is_answer_correct(orig["prediction"], orig["ground_truth"]),
                        "改进预测": str(impr["prediction"]),
                        "改进正确": self._is_answer_correct(impr["prediction"], impr["ground_truth"]),
                        "原版迭代": orig.get("iterations", 1),
                        "改进迭代": impr.get("iterations", 1),
                        "原版Token": orig.get("input_tokens", 0) + orig.get("output_tokens", 0),
                        "改进Token": impr.get("input_tokens", 0) + impr.get("output_tokens", 0),
                        "处理时间(s)": f"{orig.get('processing_time', 0):.2f}",
                        "错误模式": self.analyze_single_error_pattern(orig["prediction"], impr["prediction"],
                                                                      orig["ground_truth"])
                    })

                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name="详细结果", index=False)

            logger.info(f"Excel报告已保存: {output_path}.xlsx")

        except ImportError:
            logger.error("pandas或xlsxwriter未安装，无法生成Excel报告")
            logger.info("请运行: pip install pandas xlsxwriter")
            raise RuntimeError("缺少Excel生成依赖")
        except Exception as e:
            logger.error(f"Excel报告生成失败: {e}")
            raise RuntimeError("Excel报告生成失败")

    def analyze_single_error_pattern(self, original_pred, improved_pred, ground_truth) -> str:
        """分析单个问题的错误模式"""
        orig_correct = self._is_answer_correct(original_pred, ground_truth)
        impr_correct = self._is_answer_correct(improved_pred, ground_truth)

        if not orig_correct and impr_correct:
            return "错误→正确"
        elif orig_correct and not impr_correct:
            return "正确→错误"
        elif not orig_correct and not impr_correct:
            return "错误→错误"
        else:
            return "正确→正确"

    def create_three_method_excel(self, all_results: Dict[str, List[Dict]], output_path):
        """创建三方法对比的详细Excel报告"""
        try:
            import pandas as pd

            baseline_results = all_results["complex_cot"]
            original_results = all_results["original"]
            improved_results = all_results["improved"]

            # 分离GSM8K和MATH结果
            gsm8k_baseline = [r for r in baseline_results if r["dataset"] == "gsm8k"]
            gsm8k_original = [r for r in original_results if r["dataset"] == "gsm8k"]
            gsm8k_improved = [r for r in improved_results if r["dataset"] == "gsm8k"]

            math_baseline = [r for r in baseline_results if r["dataset"] == "math"]
            math_original = [r for r in original_results if r["dataset"] == "math"]
            math_improved = [r for r in improved_results if r["dataset"] == "math"]

            with pd.ExcelWriter(f"{output_path}.xlsx", engine='xlsxwriter') as writer:

                # 1. 总体汇总sheet
                summary_data = self._create_three_method_summary(
                    baseline_results, original_results, improved_results,
                    gsm8k_baseline, gsm8k_original, gsm8k_improved,
                    math_baseline, math_original, math_improved
                )
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="三方法汇总", index=False)

                # 2. 详细结果sheet
                detailed_data = self._create_three_method_details(
                    baseline_results, original_results, improved_results
                )
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name="详细结果", index=False)

                # 3. 错误模式分析sheet
                error_analysis = self._analyze_three_method_errors(
                    baseline_results, original_results, improved_results
                )
                error_df = pd.DataFrame(error_analysis)
                error_df.to_excel(writer, sheet_name="错误模式分析", index=False)

            logger.info(f"三方法Excel报告已保存: {output_path}.xlsx")

        except ImportError:
            logger.error("pandas或xlsxwriter未安装，无法生成Excel报告")
            raise RuntimeError("缺少Excel生成依赖")
        except Exception as e:
            logger.error(f"Excel报告生成失败: {e}")
            raise RuntimeError("Excel报告生成失败")

    def _create_three_method_summary(self, baseline_results, original_results, improved_results,
                                     gsm8k_baseline, gsm8k_original, gsm8k_improved,
                                     math_baseline, math_original, math_improved):
        """创建三方法汇总数据"""
        summary_data = []

        # 数据集级别的汇总
        for dataset_name, baseline, original, improved in [
            ("GSM8K", gsm8k_baseline, gsm8k_original, gsm8k_improved),
            ("MATH", math_baseline, math_original, math_improved),
            ("总计", baseline_results, original_results, improved_results)
        ]:
            if not baseline:  # 跳过空数据集
                continue

            # 计算各方法的准确率
            baseline_acc = self.evaluate_accuracy(baseline)
            original_acc = self.evaluate_accuracy(original)
            improved_acc = self.evaluate_accuracy(improved)

            # 计算token消耗
            baseline_tokens = self.calculate_token_consumption(baseline)
            original_tokens = self.calculate_token_consumption(original)
            improved_tokens = self.calculate_token_consumption(improved)

            summary_data.append({
                "数据集": dataset_name,
                "样本数": len(baseline),
                "基线准确率": f"{baseline_acc['accuracy']:.4f}",
                "基线正确数": baseline_acc['correct'],
                "原版准确率": f"{original_acc['accuracy']:.4f}",
                "原版正确数": original_acc['correct'],
                "改进准确率": f"{improved_acc['accuracy']:.4f}",
                "改进正确数": improved_acc['correct'],
                "原版相对基线提升": f"{original_acc['accuracy'] - baseline_acc['accuracy']:+.4f}",
                "改进相对基线提升": f"{improved_acc['accuracy'] - baseline_acc['accuracy']:+.4f}",
                "改进相对原版提升": f"{improved_acc['accuracy'] - original_acc['accuracy']:+.4f}",
                "基线Token": baseline_tokens['total_tokens'],
                "原版Token": original_tokens['total_tokens'],
                "改进Token": improved_tokens['total_tokens'],
                "基线平均迭代": f"{baseline_tokens['avg_iterations']:.2f}",
                "原版平均迭代": f"{original_tokens['avg_iterations']:.2f}",
                "改进平均迭代": f"{improved_tokens['avg_iterations']:.2f}"
            })

        return summary_data

    def _create_three_method_details(self, baseline_results, original_results, improved_results):
        """创建三方法详细结果数据"""
        detailed_data = []

        for i, (baseline, original, improved) in enumerate(zip(baseline_results, original_results, improved_results)):
            detailed_data.append({
                "序号": i + 1,
                "数据集": baseline["dataset"],
                "问题": baseline["question"][:100] + "..." if len(baseline["question"]) > 100 else baseline["question"],
                "标准答案": str(baseline["ground_truth"]),
                "基线预测": str(baseline["prediction"]),
                "基线正确": self._is_answer_correct(baseline["prediction"], baseline["ground_truth"]),
                "原版预测": str(original["prediction"]),
                "原版正确": self._is_answer_correct(original["prediction"], original["ground_truth"]),
                "改进预测": str(improved["prediction"]),
                "改进正确": self._is_answer_correct(improved["prediction"], improved["ground_truth"]),
                "基线迭代": baseline.get("iterations", 1),
                "原版迭代": original.get("iterations", 1),
                "改进迭代": improved.get("iterations", 1),
                "基线Token": baseline.get("input_tokens", 0) + baseline.get("output_tokens", 0),
                "原版Token": original.get("input_tokens", 0) + original.get("output_tokens", 0),
                "改进Token": improved.get("input_tokens", 0) + improved.get("output_tokens", 0),
                "处理时间(s)": f"{baseline.get('processing_time', 0):.2f}",
                "变化模式": self._analyze_three_method_pattern(baseline["prediction"], original["prediction"],
                                                               improved["prediction"], baseline["ground_truth"])
            })

        return detailed_data

    def _analyze_three_method_errors(self, baseline_results, original_results, improved_results):
        """分析三方法的错误模式"""
        patterns = {
            "基线→原版→改进": {"错错错": 0, "错错对": 0, "错对错": 0, "错对对": 0,
                               "对错错": 0, "对错对": 0, "对对错": 0, "对对对": 0}
        }

        for baseline, original, improved in zip(baseline_results, original_results, improved_results):
            baseline_correct = self._is_answer_correct(baseline["prediction"], baseline["ground_truth"])
            original_correct = self._is_answer_correct(original["prediction"], original["ground_truth"])
            improved_correct = self._is_answer_correct(improved["prediction"], improved["ground_truth"])

            pattern = ""
            pattern += "对" if baseline_correct else "错"
            pattern += "对" if original_correct else "错"
            pattern += "对" if improved_correct else "错"

            patterns["基线→原版→改进"][pattern] += 1

        # 转换为列表格式
        error_analysis = []
        for pattern, count in patterns["基线→原版→改进"].items():
            error_analysis.append({
                "模式": pattern,
                "数量": count,
                "描述": f"基线{pattern[0]}，原版{pattern[1]}，改进{pattern[2]}"
            })

        return error_analysis

    def _analyze_three_method_pattern(self, baseline_pred, original_pred, improved_pred, ground_truth):
        """分析单个问题的三方法变化模式"""
        baseline_correct = self._is_answer_correct(baseline_pred, ground_truth)
        original_correct = self._is_answer_correct(original_pred, ground_truth)
        improved_correct = self._is_answer_correct(improved_pred, ground_truth)

        pattern = ""
        pattern += "✓" if baseline_correct else "✗"
        pattern += "✓" if original_correct else "✗"
        pattern += "✓" if improved_correct else "✗"


        return f"{pattern[0]}→{pattern[1]}→{pattern[2]}"
