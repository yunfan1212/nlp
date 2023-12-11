#coding=utf-8
import os

def get_all_path_list(root_path):
    path_list = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for i in range(len(filenames)):
            path = os.path.join(dirpath, filenames[i])
            path_list.append(path)

    return path_list




import logging
import os.path
import sys

'''
  语料预处理：将wiki的xml转换成一行一行的text
  使用方法: 01_preprocess.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.txt
'''

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0]) #得到脚本名称
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))


    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki =WikiCorpus(inp, lemmatize=False, dictionary=[])#gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts(): #通过get_texts将维基里的每篇文章转换为一行文本，并且去掉了标点符号等内容
        output.write(space.join(text) + "\n")
        i = i+1
        if (i % 10000 == 0):
            logger.info("Saved "+str(i)+" articles.")

    output.close()
    logger.info("Finished Saved "+str(i)+" articles.")




