"""
数据加载和处理模块
整合原项目的数据处理功能，确保与论文评估标准一致
"""

import json
import re
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config):
        self.config = config

    def load_gsm8k(self, custom_test=False) -> Tuple[List[str], List[float]]:
        """
        加载GSM8K数据集
        严格按照原论文的数据处理方式
        """
        questions = []
        answers = []

        try:
            with open(self.config.GSM8K_TEST_PATH, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        questions.append(data['question'])

                        # 提取数值答案 - 与原论文完全一致
                        ans_match = re.search(r"#### ([-\d,]+)", data['answer'])
                        if ans_match:
                            ans_num = float(ans_match.group(1).replace(",", ""))
                            answers.append(ans_num)
                        else:
                            logger.warning(f"GSM8K第{line_num}行答案格式异常")
                            continue

                    except json.JSONDecodeError:
                        logger.warning(f"GSM8K第{line_num}行JSON解析失败")
                        continue
                    except Exception as e:
                        logger.warning(f"GSM8K第{line_num}行处理异常: {e}")
                        continue

        except FileNotFoundError:
            logger.error(f"GSM8K数据文件未找到: {self.config.GSM8K_TEST_PATH}")
            raise RuntimeError("GSM8K数据集加载失败")
        except Exception as e:
            logger.error(f"GSM8K数据集加载异常: {e}")
            raise RuntimeError("GSM8K数据集加载失败")

        # 如果是自定义测试，随机采样指定数量
        if custom_test:
            import random
            random.seed(self.config.RANDOM_SEED)  # 确保可重现
            sample_size = min(self.config.GSM8K_SAMPLES, len(questions))
            indices = random.sample(range(len(questions)), sample_size)
            questions = [questions[i] for i in indices]
            answers = [answers[i] for i in indices]

        logger.info(f"GSM8K数据集加载完成: {len(questions)}个问题")
        return questions, answers

    def load_math(self, custom_test=False) -> Tuple[List[str], List[str]]:
        """
        加载MATH数据集 - 支持每个学科相同数量抽样
        """
        questions = []
        answers = []

        try:
            from pathlib import Path
            import random
            math_dir = Path(self.config.MATH_TEST_PATH).parent  # datasets/math
            if not math_dir.exists():
                logger.error(f"MATH数据目录未找到: {math_dir}")
                raise RuntimeError("MATH数据集目录不存在")

            # MATH数据集按学科分类的子目录
            subject_dirs = [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
            ]

            if custom_test:
                # 自定义测试：每个学科抽取相同数量
                samples_per_subject = self.config.MATH_SAMPLES_PER_TYPE
                random.seed(self.config.RANDOM_SEED)  # 确保可重现

                logger.info(f"每个学科抽取{samples_per_subject}个问题")

                for subject in subject_dirs:
                    subject_path = math_dir / subject
                    if subject_path.exists():
                        subject_files = list(subject_path.glob("*.json"))
                        logger.debug(f"找到{subject}学科文件: {len(subject_files)}个")

                        # 每个学科随机抽取指定数量
                        if len(subject_files) >= samples_per_subject:
                            selected_files = random.sample(subject_files, samples_per_subject)
                        else:
                            selected_files = subject_files
                            logger.warning(f"{subject}学科文件不足，仅有{len(subject_files)}个")

                        for file_path in selected_files:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    questions.append(data['problem'])
                                    answers.append(self._extract_math_answer(data['solution']))
                            except Exception as e:
                                logger.warning(f"MATH文件处理异常 {file_path}: {e}")
                                continue
                    else:
                        logger.warning(f"学科目录不存在: {subject_path}")

            else:
                # 完整测试：加载所有文件
                files = []
                for subject in subject_dirs:
                    subject_path = math_dir / subject
                    if subject_path.exists():
                        subject_files = list(subject_path.glob("*.json"))
                        files.extend(subject_files)
                        logger.debug(f"找到{subject}学科文件: {len(subject_files)}个")
                    else:
                        logger.warning(f"学科目录不存在: {subject_path}")

                if not files:
                    logger.error(f"MATH数据目录为空: {math_dir}")
                    raise RuntimeError("MATH数据集为空")

                logger.info(f"MATH数据集总文件数: {len(files)}")

                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            questions.append(data['problem'])
                            answers.append(self._extract_math_answer(data['solution']))
                    except Exception as e:
                        logger.warning(f"MATH文件处理异常 {file_path}: {e}")
                        continue

        except Exception as e:
            logger.error(f"MATH数据集加载异常: {e}")
            raise RuntimeError("MATH数据集加载失败")

        logger.info(f"MATH数据集加载完成: {len(questions)}个问题")
        return questions, answers

    def _extract_math_answer(self, solution: str) -> str:
        """
        从MATH解答中提取答案
        使用与原论文相同的答案提取逻辑
        """
        if 'boxed' not in solution:
            return ""

        ans = solution.split('boxed')[-1]
        if len(ans) >= 1 and ans[0] == '{':
            stack = 1
            result = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    result += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    result += c
                else:
                    result += c
        else:
            result = ans.split('$')[0].strip()

        return self._strip_math_string(result)

    def _strip_math_string(self, s: str) -> str:
        """
        清理数学字符串
        来自原项目sc_utils.py，确保与论文评估标准一致
        """
        # 移除换行符
        s = s.replace("\n", "")
        # 移除逆空格
        s = s.replace("\\!", "")
        # 替换双反斜杠
        s = s.replace("\\\\", "\\")
        # 替换分数格式
        s = s.replace("tfrac", "frac").replace("dfrac", "frac")
        # 移除左右标记
        s = s.replace("\\left", "").replace("\\right", "")
        # 移除度数符号
        s = s.replace("^{\\circ}", "").replace("^\\circ", "")
        # 移除美元符号
        s = s.replace("\\$", "")
        # 移除百分号
        s = s.replace("\\%", "").replace("\\%", "")
        # 处理小数点
        s = s.replace(" .", " 0.").replace("{.", "{0.")
        if len(s) > 0 and s[0] == ".":
            s = "0" + s
        # 移除空格
        s = s.replace(" ", "")

        return s

    def load_prompt(self, dataset: str) -> str:
        """加载提示模板"""
        if dataset.lower() == "gsm8k":
            prompt_path = self.config.GSM8K_PROMPT_PATH
        elif dataset.lower() == "math":
            prompt_path = self.config.MATH_PROMPT_PATH
        else:
            raise ValueError(f"不支持的数据集: {dataset}")

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"提示文件未找到: {prompt_path}")
            raise RuntimeError(f"提示文件加载失败: {dataset}")

    def extract_answer_from_response(self, response: str, dataset: str) -> Any:
        """
        从模型响应中提取答案 - 严格按照原论文实现
        """
        # 删除这些调试print语句：
        # print(f"=== 完整响应 ===")
        # print(repr(response))
        # print(f"=== 响应结束 ===")

        try:
            # 按原论文extract_math_answer的逻辑处理
            if 'The answer is ' in response:
                pred = response.split('The answer is ')[-1].strip()
            elif 'the answer is ' in response:
                pred = response.split('the answer is ')[-1].strip()
            elif 'boxed' in response:
                ans = response.split('boxed')[-1]
                if len(ans) >= 1:
                    if ans[0] == '{':
                        stack = 1
                        a = ''
                        for c in ans[1:]:
                            if c == '{':
                                stack += 1
                                a += c
                            elif c == '}':
                                stack -= 1
                                if stack == 0:
                                    break
                                a += c
                            else:
                                a += c
                    else:
                        a = ans.split('$')[0].strip()
                    a = self._strip_math_string(a)
                    pred = a
                else:
                    pred = 'None'
            else:
                pattern = '-?\d*\.?\d+'
                pred = re.findall(pattern, response)
                if len(pred) >= 1:
                    pred = pred[-1]
                else:
                    pred = ''

            # 清理答案
            if pred != "":
                if pred.endswith("."):
                    pred = pred[:-1]
                if pred.endswith("/"):
                    pred = pred[:-1]

            pred = self._strip_math_string(pred)

            # 二次boxed处理（原论文重要步骤）
            if 'boxed' in pred:
                ans = pred.split('boxed')[-1]
                if len(ans) >= 1:
                    if ans[0] == '{':
                        stack = 1
                        a = ''
                        for c in ans[1:]:
                            if c == '{':
                                stack += 1
                                a += c
                            elif c == '}':
                                stack -= 1
                                if stack == 0:
                                    break
                                a += c
                            else:
                                a += c
                    else:
                        a = ans.split('$')[0].strip()
                    a = self._strip_math_string(a)
                    pred = a
                else:
                    pred = ""

            # 根据数据集类型返回
            if dataset.lower() == "gsm8k":
                if pred:
                    try:
                        result = float(pred)
                        # 删除这行：print(f"GSM8K提取成功: {result}")
                        return result
                    except:
                        # 删除这行：print(f"GSM8K转换失败: {pred}")
                        return float('nan')
                else:
                    return float('nan')
            else:  # MATH
                # 删除这行：print(f"MATH提取成功: '{pred}'")
                return pred

        except Exception as e:
            logger.warning(f"答案提取错误: {e}")
            if dataset.lower() == "gsm8k":
                return float('nan')
            else:
                return ""

    def save_results(self, results: List[Dict], output_path):
        """保存实验结果"""
        try:
            from pathlib import Path
            output_path = Path(output_path)  # 确保是Path对象
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"结果保存完成: {output_path}")
        except Exception as e:
            logger.error(f"结果保存失败: {e}")
            raise RuntimeError("结果保存失败")

    def create_batches(self, data: List, batch_size: int) -> List[List]:
        """创建数据批次以避免内存不足"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches

    def validate_data_integrity(self, questions: List, answers: List) -> bool:
        """验证数据完整性"""
        if len(questions) != len(answers):
            logger.error(f"问题和答案数量不匹配: {len(questions)} vs {len(answers)}")
            return False

        if len(questions) == 0:
            logger.error("数据集为空")
            return False

        logger.info(f"数据完整性验证通过: {len(questions)}个样本")
        return True