





'''复制已经训练过的tokenizer'''
#只需要修改一下配置文件的名字
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("../chatglm3-6b",trust_remote_code=True)
print(tokenizer.encode("白日依山尽，黄河入海流。欲穷千里目，更上一层楼。 The primary use of LLaMA is research on large language models, including"))
print(tokenizer.tokenize("白日依山尽，黄河入海流。欲穷千里目，更上一层楼。 The primary use of LLaMA is research on large language models, including"))
import os
output_dir = './transformers_tokenizer/news1/'
os.makedirs(output_dir, exist_ok=True)
# tokenizer.save_pretrained(output_dir)

#不知道什么原因，需要中tokenizer_config.py特殊字符的配置给删除掉
tokenizer1=AutoTokenizer.from_pretrained(output_dir,trust_remote_code=True)
print(tokenizer1.encode("白日依山尽，黄河入海流。欲穷千里目，更上一层楼。 The primary use of LLaMA is research on large language models, including"))
print(tokenizer1.tokenize("白日依山尽，黄河入海流。欲穷千里目，更上一层楼。 The primary use of LLaMA is research on large language models, including"))














