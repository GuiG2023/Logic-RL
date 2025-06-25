with open('eval_kk/main_eval_instruct.py', 'r') as f:
    content = f.read()

# 替换整个load_eval_records函数
old_function = '''def load_eval_records(args, subject):
    """Load evaluation records based on arguments."""
    if args.problem_type != "clean":
        records = datasets.load_dataset('K-and-K/perturbed-knights-and-knaves', 
                                        data_files=f"{args.split}/{args.problem_type}/{subject}.jsonl")["train"]
    else:
        records = datasets.load_dataset('K-and-K/knights-and-knaves', 
                                        data_files=f"{args.split}/{subject}.jsonl")["train"]
    return records'''

new_function = '''def load_eval_records(args, subject):
    """Load evaluation records from local parquet files."""
    import pandas as pd
    import os
    
    # 使用本地parquet文件
    if hasattr(args, 'data_dir') and args.data_dir:
        file_path = os.path.join(args.data_dir, "test.parquet")
    else:
        file_path = f"data/kk/instruct/{getattr(args, 'eval_nppl', 3)}ppl/test.parquet"
    
    print(f"Loading local data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    from datasets import Dataset
    records = Dataset.from_pandas(df)
    return records'''

# 执行替换
new_content = content.replace(old_function, new_function)

# 写入文件
with open('eval_kk/main_eval_instruct.py', 'w') as f:
    f.write(new_content)

print("✅ 已修复load_eval_records函数")
