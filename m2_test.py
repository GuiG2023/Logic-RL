import torch
from transformers import pipeline
import pandas as pd

print("🚀 M2上的逻辑推理测试")

# 1. 检查设备
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  设备: {device}")

# 2. 加载数据
try:
    df = pd.read_parquet('./data/kk/instruct/3ppl/test.parquet')
    print(f"📊 数据: {len(df)}行, 列: {list(df.columns)}")
    
    # 获取第一个问题
    sample = df.iloc[0]
    if hasattr(sample, 'question'):
        question = sample.question
    else:
        question = str(sample.iloc[0])  # 使用第一列
    
    print(f"❓ 问题: {question[:100]}...")
    
except Exception as e:
    print(f"❌ 数据读取失败: {e}")
    exit()

# 3. 创建模型pipeline
try:
    print("🤖 加载模型...")
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device=0 if device == "mps" else -1,
        torch_dtype=torch.float16 if device == "mps" else None
    )
    print("✅ 模型加载成功")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 4. 测试推理
try:
    prompt = f"""你是一个逻辑推理助手。请用<think>思考过程</think><answer>最终答案</answer>的格式回答。

问题: {question}"""

    print("🧠 开始推理...")
    result = generator(prompt, max_new_tokens=128, temperature=0.1)
    response = result[0]['generated_text'][len(prompt):]
    
    print(f"💭 回答: {response}")
    
    # 检查格式
    has_think = '<think>' in response and '</think>' in response
    has_answer = '<answer>' in response and '</answer>' in response
    
    print(f"\n📋 格式检查:")
    print(f"   Think标签: {'✅' if has_think else '❌'}")
    print(f"   Answer标签: {'✅' if has_answer else '❌'}")
    
    if has_think and has_answer:
        print("🎉 M2测试成功！格式正确！")
    else:
        print("⚠️  格式需要调整，但推理功能正常")
        
except Exception as e:
    print(f"❌ 推理失败: {e}")

print("\n🔄 如果成功，可以创建完整的M2版本脚本")
