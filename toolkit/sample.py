import json
import random
import argparse

def load_json(input_file):
    """从文件中读取 JSON 数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sample_data(data, num_samples, seed=None):
    """从 JSON 数据中采样指定数目的数据，并支持设置随机种子"""
    if seed is not None:
        random.seed(seed)  # 设置随机种子
    if len(data) <= num_samples:
        return data  # 如果数据量少于或等于采样数目，返回原始数据
    return random.sample(data, num_samples)

def save_json(data, output_file):
    """将采样后的 JSON 数据保存到文件中"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  # 使用 indent 格式化 JSON

def main(input_file, output_file, num_samples, seed=None):
    """主函数，控制流程"""
    # 读取 JSON 数据
    data = load_json(input_file)

    # 采样指定数量的数据，设置随机种子（如果提供）
    sampled_data = sample_data(data, num_samples, seed)

    # 保存采样后的数据到指定文件
    save_json(sampled_data, output_file)
    print(f"采样后的数据已保存到 {output_file}")

if __name__ == "__main__":
    # 使用 argparse 获取命令行参数
    
    input_file  = "/home/cjk/dj_synth_challenge/toolkit/training/data/finetuning_stage_1/mgm_instruction_stage_1.json"
    output_file = "/home/cjk/dj_synth_challenge/toolkit/training/data/finetuning_stage_1/mgm_instruction_stage_21.json"
    num_samples = 6000
    seed = 21
    # parser = argparse.ArgumentParser(description='从 JSON 文件中采样指定数量的数据并保存到另一个文件中')
    # parser.add_argument('input_file', required=False, type=str, help='输入的 JSON 文件路径', default="/home/cjk/dj_synth_challenge/toolkit/training/data/finetuning_stage_1/mgm_instruction_stage_1.json")
    # parser.add_argument('output_file', required=False, type=str, help='保存采样后 JSON 文件的路径', default="/home/cjk/dj_synth_challenge/toolkit/training/data/finetuning_stage_1/mgm_instruction_stage_2.json")
    # parser.add_argument('num_samples', required=False, type=int, help='采样的数量', default=6000)
    # parser.add_argument('--seed', required=False, type=int, default=42, help='设置随机种子（可选）')

    # args = parser.parse_args()

    # 调用主函数，传递随机种子（如果提供）
    main(input_file, output_file, num_samples, seed)