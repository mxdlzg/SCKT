from setuptools import setup, find_packages

setup(
    name='sckt',  # 你的模块名称
    version='0.1.0', # 模块版本
    packages=find_packages(), # 自动发现包含 __init__.py 的文件夹
    # 附加的元数据
    author = 'mxdlzg',
    description = 'Description of stable causal kt.'
)