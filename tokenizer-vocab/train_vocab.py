#coding=utf8


#训练分词器
def train():
    import sentencepiece as spm
    spm.SentencePieceTrainer.Train(
        input="/home/chatglm6/sentences_vocab/tianlongbabu.txt",
        model_prefix="tokenizer",
        vocab_size=50000,
        user_defined_symbols=['foo', 'bar'],
        character_coverage=1.0,
        model_type="bpe",
    )



def practice_sentencepiece():
    import sentencepiece as spm
    sp_model=spm.SentencePieceProcessor()
    sp_model.Load("./tokenizer.model")
    #编码
    print(sp_model.EncodeAsPieces("你好是一个汉语词语"))
    print(sp_model.EncodeAsIds("你好是一个汉语词语"))
    #解码
    print(sp_model.DecodePieces(['▁', '你好', '是', '一个', '汉', '语', '词', '语']))
    print(sp_model.Decode([46706, 2382, 46699, 21, 47120, 47105, 48432, 47105]))




#practice_sentencepiece()
#transformer 分词器使用 加载
def tf_practice():

    import os
    from transformers import LlamaTokenizer
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    import sentencepiece as spm

    chinese_sp_model_file = "./tokenizer.model"

    # 加载分词器模型
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)

    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())   #序列化

    ## Save
    output_dir = './transformers_tokenizer/chinese/'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + 'chinese.model', 'wb') as f:
        f.write(chinese_spm.SerializeToString())        #模型保存


    #transformer 加载
    tokenizer = LlamaTokenizer(vocab_file=output_dir + 'chinese.model')
    tokenizer.save_pretrained(output_dir)                          #并保存到transformer格式

    #测试
    chinese_tokenizer = LlamaTokenizer.from_pretrained(output_dir)    #transformer 格式加载
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_tokenizer.tokenize(text)}")


#tf_practice()

#transformer 分词器合并

def combine():
    import os
    from transformers import LlamaTokenizer
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    import sentencepiece as spm

    llama_tokenizer_dir = "/home/llama_/CodeLlama-7b/tokenizer.model"
    chinese_sp_model_file = "./tokenizer.model"

    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)    #已有分词器

    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file)    #新分词器

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())   #已有分词序列化

    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())    #现有分词序列化

    # print number of tokens
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    ## Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)    #分词片段
    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    for p in chinese_spm.pieces:                         #将中文分词添加到已有分词器中
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()   #创建词片对象
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)   #添加到词片段集合中
    print(f"New model pieces: {len(llama_spm.pieces)}")

    ## Save 保存
    output_sp_dir = 'transformers_tokenizer/llama_chinese'
    output_hf_dir = 'transformers_tokenizer/llama_chinese'  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/chinese_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())                     #序列号，并保存到新模型

    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama.model')   #加载新模型，并保存
    tokenizer.save_pretrained(output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")

#combine()

#单独使用合并后的transformers
def test():
    from transformers import LlamaTokenizer
    output_hf_dir = 'transformers_tokenizer/llama_chinese'

    tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
       The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print(f"Tokenized by Chinese-LLaMA tokenizer:{tokenizer.tokenize(text)}")

