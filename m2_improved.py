import torch
from transformers import pipeline
import pandas as pd

print("🚀 M2改进版测试")

# 设备检测
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  设备: {device}")

# 加载数据
df = pd.read_parquet('./data/kk/instruct/3ppl/test.parquet')
question = str(df.iloc[0]['quiz'])  # 使用quiz列
print(f"❓ 问题: {question[:100]}...")

# 加载模型
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0 if device == "mps" else -1,
    torch_dtype=torch.float16 if device == "mps" else None
)

# 使用更强的格式要求
prompt = f"""<|im_start|>system
You MUST respond using this exact format:
<think>
Your reasoning process here
</think>
<answer>
Your final answer here
</answer>

IMPORTANT: Always include both <think></think> and <answer></answer> tags.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>"""

print("🧠 开始推理...")
result = generator(
    prompt, 
    max_new_tokens=256, 
    temperature=0.1,
    do_sample=False,  # 确定性生成
    pad_token_id=generator.tokenizer.eos_token_id
)

response = result[0]['generated_text'][len(prompt):]
print(f"💭 回答: {response}")

# 检查格式
has_think = '</think>' in response
has_answer = '<answer>' in response and '</answer>' in response
format_correct = has_think and has_answer

print(f"\n📋 格式检查:")
print(f"   Think标签: {'✅' if has_think else '❌'}")
print(f"   Answer标签: {'✅' if has_answer else '❌'}")
print(f"   格式正确: {'✅' if format_correct else '❌'}")

if format_correct:
    print("🎉 完美！M2测试成功，格式正确！")
    print("✈️  可以转到Colab做大规模实验了")
else:
    print("📝 需要进一步调整prompt，但功能正常")
    
    # 尝试解析结果
    if '</think>' in response:
        think_part = response.split('</think>')[0]
        print(f"🧠 Think部分: {think_part[:100]}...")
    
