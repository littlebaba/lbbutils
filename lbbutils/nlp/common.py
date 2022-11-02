import os
import sys
import tarfile
import urllib

import dill as pickle
import spacy
import torchtext


def file_exist(dir_name, file_name):
    """
    文件是否存在
    Args:
        dir_name:
        file_name:

    Returns:

    """
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def _download_file(download_dir, url: str):
    """
    下载文件
    Args:
        download_dir:
        url:

    Returns:返回文件名

    """
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"已下载：{url}（文件位于{filename}）。\n")
    else:
        sys.stderr.write(f"正在从{url}下载文件{filename}.\n")
        urllib.request.urlretrieve(url, filename=filename, reporthook=None)
    return filename


def download_and_extract(download_dir, url, src_filename, trg_filename):
    """

    Args:
        download_dir:
        url:
        src_filename:
        trg_filename:

    Returns:返回源文件和目标文件路径

    """
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"Aleady downloaded and extracted {url}.\n")
        return src_path, trg_path
    compressed_file = _download_file(download_dir, url)
    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(corpus_tar, download_dir)
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path


def get_raw_files(raw_dir, sources):
    """

    Args:
        raw_dir:
        sources:

    Returns:以字典的形式返回源文件和目标文件

    """
    raw_files = {"src": [], "trg": []}
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d["url"], d["src"], d["trg"])
        raw_dir["src"].append(src_file)
        raw_dir["trg"].append(trg_file)
    return raw_files


def complie_files(raw_dir, raw_files, prefix):
    '''
    文件压缩，其实是将一种文件压缩成另一文件格式
    Args:
        raw_dir:
        raw_files:
        prefix:

    Returns:

    '''
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found,skip the merging process.\n")
        return src_fpath, trg_fpath
    sys.stderr.write(f"Merged files into two files:{src_fpath} and {trg_fpath}.\n")
    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f' Input files: -SRC:{src_inf}, and -TRG:{trg_inf}')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for i, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_inf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, "Number of lines in two files are inconsistent."
    return src_fpath, trg_fpath


def mkdir_if_needed(dir_name):
    '''
    如果目录不存在，创建目录
    Args:
        dir_name:

    Returns:

    '''
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def main_WMT16():
    src_lang_model = spacy.load('de_core_news_sm')
    trg_lang_model = spacy.load('en_core_web_sm')

    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    SRC = torchtext.legacy.data.Field(tokenize=tokenize_src, lower=True, pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)
    TRG = torchtext.legacy.data.Field(tokenize=tokenize_trg, lower=True, pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    MAX_LEN = 100
    MIN_FREQ = 3

    def filter_examples_with_length(x):
        return len(vars(x)['src'] <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

    # train, val, test = torchtext.datasets.Multi30k.split(
    #     exts=('.' + 'de', '.' + 'en'),
    #     fields=(SRC, TRG),
    #     filter_pred=filter_examples_with_length
    # )

    train, val, test = torchtext.datasets.Multi30k(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))

    # SRC.build_vocab(train.src,min_freq=MIN_FREQ)
    # TRG.build_vocab(train.trg,min_freq=MIN_FREQ)

    data = {
        # 'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': val.examples
    }
    print('[info] Dumping the processed data to pickle file', "m30k_deen_shr.pkl")
    pickle.dump(data, open('m30k_deen_shr.pkl', 'wb'))





if __name__ == '__main__':
    _main_WMT16()
