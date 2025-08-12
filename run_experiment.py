#!/usr/bin/env python3
"""
LECO实验主脚本
实现严格的科学实验流程，确保结果的可重现性和可靠性
增加断点续传和结果验证功能
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from tqdm import tqdm

from config import Config
from core.data_handler import DataHandler
from core.model_manager import ModelManager
from core.confidence_calc import ConfidenceCalculator
from core.error_detection import ErrorDetector
from core.evaluator import Evaluator
from typing import Dict

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 改为WARNING
    format='%(levelname)s - %(message)s',  # 简化格式
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LECOExperiment:
    def __init__(self, config):
        self.config = config
        self.data_handler = DataHandler(config)
        self.model_manager = ModelManager(config)
        self.confidence_calc = ConfidenceCalculator(config)
        self.error_detector = ErrorDetector(config)
        self.evaluator = Evaluator(config)

    def setup(self):
        """初始化实验环境"""
        logger.info("初始化LECO实验环境")

        try:
            # 设置随机种子确保可重现性
            self.config.set_random_seed()
            logger.info(f"随机种子设置为: {self.config.RANDOM_SEED}")

            # 确保目录存在
            self.config.ensure_dirs()

            # 加载模型
            self.model_manager.load_model()

            # 验证模型健康状态
            if not self.model_manager.health_check():
                raise RuntimeError("模型健康检查失败")

            logger.info("实验环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"实验环境初始化失败: {e}")
            return False

    def run_combined_experiment(self, quick_test: bool = False, custom_test: bool = False):
        """运行GSM8K+MATH的三方法对比实验 - 加强内存管理版本"""
        test_mode = "custom" if custom_test else ("quick" if quick_test else "full")
        logger.info(f"开始GSM8K+MATH三方法对比实验 (测试模式: {test_mode})")

        try:
            # 加载两个数据集（使用固定随机种子确保一致性）
            gsm8k_questions, gsm8k_answers = self.data_handler.load_gsm8k(custom_test)
            math_questions, math_answers = self.data_handler.load_math(custom_test if not quick_test else True)

            if quick_test:
                gsm8k_questions = gsm8k_questions[:5]
                gsm8k_answers = gsm8k_answers[:5]
                math_questions = math_questions[:5]
                math_answers = math_answers[:5]

            # 组合所有问题（固定顺序）
            all_questions = []
            all_answers = []
            all_datasets = []

            for q, a in zip(gsm8k_questions, gsm8k_answers):
                all_questions.append(q)
                all_answers.append(a)
                all_datasets.append("gsm8k")

            for q, a in zip(math_questions, math_answers):
                all_questions.append(q)
                all_answers.append(a)
                all_datasets.append("math")

            total_questions = len(all_questions)
            logger.info(f"总共{total_questions}个问题 (GSM8K: {len(gsm8k_questions)}, MATH: {len(math_questions)})")

            # 三方法对比实验 - 加强内存管理
            methods = ["complex_cot", "original", "improved"]
            all_results = {method: [] for method in methods}

            with tqdm(total=total_questions * 3, desc="三方法对比") as pbar:
                for method_idx, method in enumerate(methods):
                    pbar.set_description(f"{method}方法")

                    # 每个方法开始前强制清理内存
                    import gc
                    self.model_manager.clear_memory()
                    gc.collect()

                    for i, (question, answer, dataset) in enumerate(zip(all_questions, all_answers, all_datasets)):
                        try:
                            result = self.run_single_question(question, answer, dataset, method)
                            all_results[method].append(result)

                            # 每个问题完成后都清理内存
                            self.model_manager.clear_memory()

                            # 每3题强制深度清理
                            if (i + 1) % 3 == 0:
                                gc.collect()
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()

                            pbar.update(1)

                        except Exception as e:
                            logger.error(f"第{i + 1}题处理失败 (方法: {method}): {e}")
                            # 添加错误结果，确保结果列表长度一致
                            error_result = {
                                "question": question,
                                "ground_truth": answer,
                                "prediction": None,
                                "answer_candidates": [],
                                "iterations": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "step_info": [],
                                "iteration_details": [],
                                "processing_time": 0,
                                "method": method,
                                "dataset": dataset,
                                "error": str(e)
                            }
                            all_results[method].append(error_result)
                            pbar.update(1)

                            # 错误后强制清理
                            self.model_manager.clear_memory()
                            gc.collect()

                    # 每个方法完成后，检查结果数量一致性
                    logger.info(f"{method}方法完成，处理了{len(all_results[method])}个问题")

            # 验证结果完整性
            expected_count = total_questions
            for method in methods:
                actual_count = len(all_results[method])
                if actual_count != expected_count:
                    logger.error(f"{method}方法结果数量不匹配: 期望{expected_count}, 实际{actual_count}")
                    # 补齐缺失的结果
                    while len(all_results[method]) < expected_count:
                        missing_idx = len(all_results[method])
                        all_results[method].append({
                            "question": all_questions[missing_idx] if missing_idx < len(all_questions) else "Unknown",
                            "ground_truth": all_answers[missing_idx] if missing_idx < len(all_answers) else "Unknown",
                            "prediction": None,
                            "answer_candidates": [],
                            "iterations": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "step_info": [],
                            "iteration_details": [],
                            "processing_time": 0,
                            "method": method,
                            "dataset": all_datasets[missing_idx] if missing_idx < len(all_datasets) else "unknown",
                            "error": "Missing result"
                        })

            # 生成三方法对比Excel报告
            report_path = self.config.RESULTS_DIR / f"LECO_three_method_comparison_{test_mode}"
            self.evaluator.create_three_method_excel(all_results, report_path)

            logger.info(f"三方法对比实验完成，Excel报告保存至: {report_path}.xlsx")
            return all_results

        except Exception as e:
            logger.error(f"三方法对比实验失败: {e}")
            raise

    def save_checkpoint(self, results: list, batch_idx: int, dataset: str, method: str):
        """保存实验检查点"""
        try:
            checkpoint_data = {
                "batch_idx": batch_idx,
                "results": results,
                "timestamp": time.time(),
                "dataset": dataset,
                "method": method,
                "random_seed": self.config.RANDOM_SEED
            }

            checkpoint_file = self.config.get_checkpoint_file(dataset, method, batch_idx)
            checkpoint_file.parent.mkdir(exist_ok=True, parents=True)

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"检查点已保存: batch {batch_idx}")

        except Exception as e:
            logger.error(f"检查点保存失败: {e}")

    def load_checkpoint(self, dataset: str, method: str) -> tuple:
        """加载最新的检查点"""
        try:
            checkpoint_dir = self.config.CHECKPOINTS_DIR
            pattern = f"checkpoint_{dataset}_{method}_batch*.json"

            checkpoint_files = list(checkpoint_dir.glob(pattern))
            if not checkpoint_files:
                return [], 0

            # 找到最新的检查点
            latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)

            with open(latest_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            logger.info(f"从检查点恢复: batch {checkpoint_data['batch_idx']}")
            return checkpoint_data['results'], checkpoint_data['batch_idx']

        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return [], 0

    def run_single_question(self, question: str, ground_truth, dataset: str,
                            method: str = "original") -> dict:
        """处理单个问题 - 简化日志版本"""

        # 添加对complex_cot的支持
        if method == "complex_cot":
            return self.run_complex_cot(question, ground_truth, dataset)

        # 确保start_time在所有路径中都被定义
        start_time = time.time()

        try:
            # 加载提示模板
            prompt_template = self.data_handler.load_prompt(dataset)

            # 构建完整提示
            if dataset.lower() == "gsm8k":
                full_prompt = f"{prompt_template}\n\nQuestion: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
                max_tokens = self.config.MAX_NEW_TOKENS
            else:  # MATH
                full_prompt = f"{prompt_template}\n\nQuestion: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nA:"
                max_tokens = self.config.MAX_NEW_TOKENS_MATH

            # 记录token使用
            total_input_tokens = 0
            total_output_tokens = 0

            # 初始推理
            generated_text, token_ids, token_probs = self.model_manager.generate_with_logprobs(
                full_prompt, max_tokens
            )

            input_tokens = len(self.model_manager.tokenizer.encode(full_prompt))
            output_tokens = len(token_ids)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # 提取步骤信息和置信度
            step_info = self.confidence_calc.extract_step_info(
                token_ids, token_probs, generated_text, self.model_manager.tokenizer
            )

            # 提取初始答案
            initial_answer = self.data_handler.extract_answer_from_response(generated_text, dataset)
            answer_candidates = [initial_answer]

            # LECO迭代过程
            current_solution_steps = [step["sentence"] for step in step_info]
            iteration_num = 0
            prev_answer = "NULL"
            iteration_details = []

            while self.error_detector.should_continue_iteration(prev_answer, initial_answer, iteration_num):
                prev_answer = initial_answer
                iteration_start = time.time()

                # 选择错误检测方法
                if method == "improved":
                    error_step_idx, potential_errors = self.error_detector.find_error_step_improved(step_info)
                else:
                    error_step_idx, potential_errors = self.error_detector.find_error_step_original(step_info)

                # 如果错误步骤是最后一步，跳出循环
                if error_step_idx >= len(current_solution_steps) - 1:
                    break

                # 构建重新思考的提示
                correct_steps = "\n".join(current_solution_steps[:error_step_idx])

                if dataset.lower() == "gsm8k":
                    rethink_prompt = f"{prompt_template}\n\nQuestion: {question}\nAnswer: {correct_steps}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                else:
                    rethink_prompt = f"{prompt_template}\n\nQuestion: {question}\nA: {correct_steps}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

                # 重新生成
                generated_text, token_ids, token_probs = self.model_manager.generate_with_logprobs(
                    rethink_prompt, max_tokens
                )

                input_tokens = len(self.model_manager.tokenizer.encode(rethink_prompt))
                output_tokens = len(token_ids)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # 提取新的步骤信息
                new_step_info = self.confidence_calc.extract_step_info(
                    token_ids, token_probs, generated_text, self.model_manager.tokenizer
                )

                # 更新解决方案
                current_solution_steps = current_solution_steps[:error_step_idx] + \
                                         [step["sentence"] for step in new_step_info]
                step_info = step_info[:error_step_idx] + new_step_info

                # 提取新答案
                new_answer = self.data_handler.extract_answer_from_response(generated_text, dataset)
                answer_candidates.append(new_answer)
                initial_answer = new_answer

                # 记录迭代详情
                iteration_details.append({
                    "iteration": iteration_num,
                    "error_step_idx": error_step_idx,
                    "error_confidence": potential_errors[0]["total_confidence"] if potential_errors else 0.0,
                    "new_answer": new_answer,
                    "time_taken": time.time() - iteration_start
                })

                iteration_num += 1

            # 选择最终答案（多数投票）
            final_answer = self.error_detector.majority_vote(answer_candidates)
            processing_time = time.time() - start_time

            return {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": final_answer,
                "answer_candidates": answer_candidates,
                "iterations": iteration_num + 1,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "step_info": step_info,
                "iteration_details": iteration_details,
                "processing_time": processing_time,
                "method": method,
                "dataset": dataset
            }

        except Exception as e:
            logger.error(f"问题处理失败: {e}")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": None,
                "answer_candidates": [],
                "iterations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "step_info": [],
                "iteration_details": [],
                "processing_time": time.time() - start_time,  # 确保这里能访问到start_time
                "method": method,
                "dataset": dataset,
                "error": str(e)
            }

    def run_dataset(self, dataset: str, method: str = "original", quick_test: bool = False, custom_test: bool = False):
        """运行完整数据集实验 """
        test_mode = "custom" if custom_test else ("quick" if quick_test else "full")
        logger.info(f"开始{dataset}数据集实验 (方法: {method}, 测试模式: {test_mode})")

        try:
            # 尝试从检查点恢复
            existing_results, start_batch = self.load_checkpoint(dataset, method)
            if existing_results:
                logger.info(f"从检查点恢复，已完成{len(existing_results)}个问题")

            # 加载数据 - 根据测试模式选择加载方式
            if dataset.lower() == "gsm8k":
                if quick_test:
                    questions, answers = self.data_handler.load_gsm8k(False)
                    questions = questions[:self.config.QUICK_TEST_SIZE]
                    answers = answers[:self.config.QUICK_TEST_SIZE]
                elif custom_test:
                    questions, answers = self.data_handler.load_gsm8k(True)  # 自定义测试
                else:
                    questions, answers = self.data_handler.load_gsm8k(False)  # 完整测试
            elif dataset.lower() == "math":
                if quick_test:
                    questions, answers = self.data_handler.load_math(False)
                    questions = questions[:self.config.QUICK_TEST_SIZE]
                    answers = answers[:self.config.QUICK_TEST_SIZE]
                elif custom_test:
                    questions, answers = self.data_handler.load_math(True)  # 自定义测试
                else:
                    questions, answers = self.data_handler.load_math(False)  # 完整测试
            else:
                raise ValueError(f"不支持的数据集: {dataset}")

            # 验证数据完整性
            if not self.data_handler.validate_data_integrity(questions, answers):
                raise RuntimeError("数据完整性验证失败")

            # 跳过已完成的部分
            if existing_results:
                questions = questions[len(existing_results):]
                answers = answers[len(existing_results):]
                results = existing_results
            else:
                results = []

            # 分批处理
            batches = self.data_handler.create_batches(
                list(zip(questions, answers)),
                self.config.BATCH_SIZE
            )

            for batch_idx, batch in enumerate(tqdm(batches, desc=f"处理{dataset}批次")):
                batch_results = []

                for question, answer in tqdm(batch, desc="问题", leave=False):
                    try:
                        result = self.run_single_question(question, answer, dataset, method)
                        batch_results.append(result)

                        # 检查资源使用
                        warnings = self.config.check_resources()
                        if warnings:
                            for warning in warnings:
                                logger.warning(warning)

                            # 强制内存清理
                            self.model_manager.clear_memory()

                    except Exception as e:
                        logger.error(f"问题处理失败: {e}")
                        # 记录失败的问题
                        batch_results.append({
                            "question": question,
                            "ground_truth": answer,
                            "prediction": None,
                            "error": str(e)
                        })

                results.extend(batch_results)

                # 定期保存检查点
                if (batch_idx + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(results, start_batch + batch_idx + 1, dataset, method)

            # 验证结果与论文基准
            if not quick_test and method == "original":
                accuracy_metrics = self.evaluator.evaluate_accuracy(results)
                self.config.validate_results_against_paper(
                    accuracy_metrics["accuracy"] * 100,
                    dataset
                )

            return results

        except Exception as e:
            logger.error(f"数据集实验失败: {e}")
            raise

    def run_comparison_experiment(self, dataset: str, quick_test: bool = False, custom_test: bool = False):
        """运行对比实验（原版 vs 改进版）"""
        test_mode = "custom" if custom_test else ("quick" if quick_test else "full")
        logger.info(f"开始{dataset}对比实验 (测试模式: {test_mode})")

        try:
            # 运行原版LECO
            logger.info("运行原版LECO方法")
            original_results = self.run_dataset(dataset, "original", quick_test, custom_test)

            # 保存原版结果
            original_output = self.config.get_output_file(dataset, "original", test_mode)
            self.data_handler.save_results(original_results, original_output)

            # 运行改进版LECO
            logger.info("运行改进版LECO方法")
            improved_results = self.run_dataset(dataset, "improved", quick_test, custom_test)

            # 保存改进版结果
            improved_output = self.config.get_output_file(dataset, "improved", test_mode)
            self.data_handler.save_results(improved_results, improved_output)

            # 生成对比报告
            logger.info("生成对比报告")
            comparison_report = self.evaluator.generate_comparison_report(
                original_results, improved_results, dataset
            )

            # 保存报告
            report_path = self.config.RESULTS_DIR / f"{dataset}_comparison_report_{test_mode}"
            self.evaluator.save_report(comparison_report, report_path)

            # 创建可视化
            try:
                self.evaluator.create_visualization(comparison_report, self.config.RESULTS_DIR)
            except Exception as e:
                logger.warning(f"可视化创建失败: {e}")

            # 打印简要结果
            self._print_summary_results(comparison_report)

            return comparison_report

        except Exception as e:
            logger.error(f"对比实验失败: {e}")
            raise

    def _print_combined_summary(self, summary: Dict):
        """打印两个数据集的汇总结果"""
        print("\n" + "=" * 80)
        print("GSM8K + MATH 联合实验汇总报告")
        print("=" * 80)
        print(f"测试模式: {summary['experiment_info']['test_mode']}")
        print(f"总样本数: {summary['experiment_info']['total_samples']}")
        print("-" * 80)

        # 各数据集结果
        for dataset, results in summary["dataset_results"].items():
            print(f"{dataset.upper()}数据集:")
            print(f"  原版准确率: {results['original_accuracy']:.4f}")
            print(f"  改进准确率: {results['improved_accuracy']:.4f}")
            print(f"  准确率提升: {results['improvement']:+.4f}")
            print(f"  样本数: {results['sample_size']}")
            print()

        # 整体改进
        overall = summary["overall_improvement"]
        print(f"整体改进效果:")
        print(f"  原版总准确率: {overall['original_accuracy']:.4f}")
        print(f"  改进总准确率: {overall['improved_accuracy']:.4f}")
        print(f"  总体准确率提升: {overall['accuracy_gain']:+.4f}")
        print(f"  Token节省: {overall['token_reduction']:+,}")
        print("=" * 80)

    def _print_summary_results(self, report: dict):
        """打印实验结果摘要"""
        print("\n" + "=" * 70)
        print(f"LECO方法对比实验结果 - {report['dataset'].upper()}")
        print("=" * 70)
        print(f"测试模式: {report['experiment_info']['test_type']}")
        print(f"样本数量: {report['experiment_info']['sample_size']}")
        print(f"随机种子: {report['experiment_info']['random_seed']}")
        print("-" * 70)

        # 准确率对比
        orig_acc = report['original_method']['accuracy']
        impr_acc = report['improved_method']['accuracy']
        orig_correct = report['original_method']['correct']
        orig_total = report['original_method']['total']
        impr_correct = report['improved_method']['correct']
        impr_total = report['improved_method']['total']

        print(f"原版LECO准确率: {orig_acc:.4f} ({orig_correct}/{orig_total})")
        print(f"改进版本准确率: {impr_acc:.4f} ({impr_correct}/{impr_total})")
        print(
            f"准确率提升: {report['improvement']['accuracy_gain']:+.4f} ({report['improvement']['accuracy_gain_percent']:+.2f}%)")

        print("-" * 70)

        # Token消耗对比
        orig_tokens = report['original_method']['token_consumption']['total_tokens']
        impr_tokens = report['improved_method']['token_consumption']['total_tokens']
        print(f"原版Token消耗: {orig_tokens:,}")
        print(f"改进版Token消耗: {impr_tokens:,}")
        print(
            f"Token变化: {report['improvement']['token_reduction']:+,} ({report['improvement']['token_reduction_percent']:+.2f}%)")

        print("-" * 70)

        # 迭代次数
        orig_iter = report['original_method']['token_consumption']['avg_iterations']
        impr_iter = report['improved_method']['token_consumption']['avg_iterations']
        print(f"平均迭代次数: {orig_iter:.2f} → {impr_iter:.2f} ({report['improvement']['iteration_reduction']:+.2f})")

        # 错误模式分析
        patterns = report['error_patterns']
        total_changes = patterns['W2R'] + patterns['R2W'] + patterns['W2W']
        if total_changes > 0:
            print("-" * 70)
            print("错误模式变化:")
            print(f"  错误→正确: {patterns['W2R']} ({patterns['W2R'] / total_changes * 100:.1f}%)")
            print(f"  正确→错误: {patterns['R2W']} ({patterns['R2W'] / total_changes * 100:.1f}%)")
            print(f"  错误→错误: {patterns['W2W']} ({patterns['W2W'] / total_changes * 100:.1f}%)")

        # 统计显著性
        if report.get('statistical_analysis'):
            stats = report['statistical_analysis']
            print("-" * 70)
            print(f"统计显著性: {'显著' if stats.get('improvement_significant', False) else '不显著'}")
            if isinstance(stats.get('p_value'), (int, float)):
                print(f"p值: {stats['p_value']:.4f}")

        print("=" * 70)

    def run_both_datasets_experiment(self, quick_test: bool = False, custom_test: bool = False):
        """运行两个数据集的完整对比实验"""
        test_mode = "custom" if custom_test else ("quick" if quick_test else "full")
        logger.info(f"开始GSM8K + MATH联合实验 (测试模式: {test_mode})")

        all_results = {}

        try:
            # 数据集列表
            datasets = ["gsm8k", "math"]

            for dataset in datasets:
                logger.info(f"开始处理数据集: {dataset}")

                # 运行原版LECO
                logger.info(f"运行{dataset}原版LECO方法")
                original_results = self.run_dataset(dataset, "original", quick_test, custom_test)

                # 运行改进版LECO
                logger.info(f"运行{dataset}改进版LECO方法")
                improved_results = self.run_dataset(dataset, "improved", quick_test, custom_test)

                # 保存单个数据集结果
                original_output = self.config.get_output_file(dataset, "original", test_mode)
                improved_output = self.config.get_output_file(dataset, "improved", test_mode)
                self.data_handler.save_results(original_results, original_output)
                self.data_handler.save_results(improved_results, improved_output)

                # 生成单个数据集的对比报告
                logger.info(f"生成{dataset}对比报告")
                comparison_report = self.evaluator.generate_comparison_report(
                    original_results, improved_results, dataset
                )

                # 保存单个数据集报告
                report_path = self.config.RESULTS_DIR / f"{dataset}_comparison_report_{test_mode}"
                self.evaluator.save_report(comparison_report, report_path)

                # 存储结果用于汇总
                all_results[dataset] = {
                    "original": original_results,
                    "improved": improved_results,
                    "comparison": comparison_report
                }

                # 打印单个数据集结果
                self._print_summary_results(comparison_report)

            # 生成汇总报告
            logger.info("生成两个数据集汇总报告")
            summary_report = self.evaluator.generate_combined_summary(all_results, test_mode)

            # 保存汇总报告
            summary_path = self.config.RESULTS_DIR / f"combined_summary_report_{test_mode}"
            self.evaluator.save_report(summary_report, summary_path)

            # 打印汇总结果
            self._print_combined_summary(summary_report)

            return all_results

        except Exception as e:
            logger.error(f"联合实验失败: {e}")
            raise

    def run_complex_cot(self, question: str, ground_truth, dataset: str) -> dict:
        """
        基线方法：Complex CoT - 单次推理，不迭代
        """
        start_time = time.time()

        try:
            # 加载提示模板（与LECO相同）
            prompt_template = self.data_handler.load_prompt(dataset)

            # 构建完整提示
            if dataset.lower() == "gsm8k":
                full_prompt = f"{prompt_template}\n\nQuestion: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
                max_tokens = self.config.MAX_NEW_TOKENS
            else:  # MATH
                full_prompt = f"{prompt_template}\n\nQuestion: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nA:"
                max_tokens = self.config.MAX_NEW_TOKENS_MATH

            # 单次生成（关键：不迭代）
            generated_text, token_ids, token_probs = self.model_manager.generate_with_logprobs(
                full_prompt, max_tokens
            )

            input_tokens = len(self.model_manager.tokenizer.encode(full_prompt))
            output_tokens = len(token_ids)

            # 提取步骤信息（用于统计，但不用于错误检测）
            step_info = self.confidence_calc.extract_step_info(
                token_ids, token_probs, generated_text, self.model_manager.tokenizer
            )

            # 提取答案
            final_answer = self.data_handler.extract_answer_from_response(generated_text, dataset)
            processing_time = time.time() - start_time

            return {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": final_answer,
                "answer_candidates": [final_answer],
                "iterations": 1,  # 只有1次迭代
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "step_info": step_info,
                "iteration_details": [],
                "processing_time": processing_time,
                "method": "complex_cot",
                "dataset": dataset
            }

        except Exception as e:
            logger.error(f"Complex CoT处理失败: {e}")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": None,
                "answer_candidates": [],
                "iterations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "step_info": [],
                "iteration_details": [],
                "processing_time": 0,
                "method": "complex_cot",
                "dataset": dataset,
                "error": str(e)
            }


def main():
    try:
        parser = argparse.ArgumentParser(description="LECO三方法对比实验")
        parser.add_argument("--quick_test", action="store_true", help="快速测试模式(15题)")
        parser.add_argument("--custom_test", action="store_true", help="自定义测试模式(360题)")

        args = parser.parse_args()

        # 验证环境
        issues = Config.validate_environment()
        if issues:
            print("环境问题:")
            for issue in issues:
                print(f"  - {issue}")
            user_input = input("是否继续实验？(y/N): ")
            if user_input.lower() != 'y':
                sys.exit(1)

        # 运行三方法对比实验
        experiment = LECOExperiment(Config)
        if not experiment.setup():
            logger.error("实验环境初始化失败")
            sys.exit(1)

        experiment.run_combined_experiment(args.quick_test, args.custom_test)
        print("三方法对比实验成功完成")

    except KeyboardInterrupt:
        print("实验被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"实验失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()