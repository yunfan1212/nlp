

def make_self_tokenizer():
    import os
    from sentences_vocab.tokenization import NewChineseTokenizer
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    chinese_sp_model_file = "./tokenizer.model"

    # 加载分词器模型
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())  # 序列化

    ## Save
    output_dir = './transformers_tokenizer/news/'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + 'chinese.model', 'wb') as f:
        f.write(chinese_spm.SerializeToString())  # 模型保存

    #加载
    tokenizer=NewChineseTokenizer(vocab_file=output_dir+"chinese.model")
    tokenizer.save_pretrained(output_dir)  # 并保存到transformer格式

#make_self_tokenizer()


def test():
    #手动导入类，加载
    from sentences_vocab.tokenization import NewChineseTokenizer
    output_dir = './transformers_tokenizer/news/'
    chinese_tokenizer=NewChineseTokenizer.from_pretrained(output_dir)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
        The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_tokenizer.tokenize(text)}")

test()   #创建自己的tokenizer
print("======================================")

def test1():
    '''
    修改配置文件，让模型能够找到对应的类
            {
          "name_or_path": "",
          "remove_space": false,
          "do_lower_case": false,
          "tokenizer_class": "ChatGLMTokenizer",
          "auto_map": {
            "AutoTokenizer": [
              "tokenization_chatglm.ChatGLMTokenizer",
              null
              ]
               }
            }
    开始：
        {
      "add_bos_token": true,
      "add_eos_token": false,
      "added_tokens_decoder": {
       ...
      "tokenizer_class": "NewChineseTokenizer",
      "unk_token": "<unk>",
         }
    插入后：
     {
      "add_bos_token": true,
      "add_eos_token": false,
      "added_tokens_decoder": {
       ...
      "tokenizer_class": "NewChineseTokenizer",
      "unk_token": "<unk>",

      "name_or_path": "",
      "remove_space": false,
      "do_lower_case": false,
      "tokenizer_class": "ChatGLMTokenizer",
      "auto_map": {
        "AutoTokenizer": [
          "tokenization_chatglm.ChatGLMTokenizer",
          null
          ]
           }

         }
    '''
    #自动加载
    from transformers import AutoTokenizer
    output_dir = './transformers_tokenizer/news/'
    chinese_tokenizer=AutoTokenizer.from_pretrained(output_dir,trust_remote_code=True)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
        The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_tokenizer.tokenize(text)}")

#test1()




