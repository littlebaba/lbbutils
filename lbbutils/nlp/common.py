import os
import sys
import urllib


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


def download_file(download_dir, url: str):
    """
    下载文件
    Args:
        download_dir:
        url:

    Returns:

    """
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"已下载：{url}（文件位于{filename}）。\n")
    else:
        sys.stderr.write(f"正在从{url}下载文件{filename}.\n")
        urllib.request.urlretrieve(url, filename=filename, reporthook=None)
    return filename
