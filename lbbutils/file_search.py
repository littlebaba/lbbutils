# @Author  : liheng
# @Time    : 2020/8/8 18:58
"""
实现文本搜索
    如果一个文件中有搜索内容的话打开
    如果没有搜索内容的话关闭s
"""
import re


def has_key(path, key=None):
    with open(path, 'r',encoding='utf8') as f:
        context = f.readlines()
    for k,c in enumerate(context):
        match=re.search(key, c)
        if match is not None:
            print(f'{key} 出现在文件中第{k}行,{[col[0] for col in match.regs][0]}列')


if __name__ == '__main__':
    has_key("./timer.py","fnc")
