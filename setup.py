from setuptools import setup, find_packages

def get_requirements(filename):
    """读取依赖文件并返回依赖列表"""
    with open(f'requirements/{filename}') as f:
        # 过滤掉注释和空行，以及 -r base.txt
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]

def get_all_requirements():
    """获取所有依赖"""
    base = get_requirements('base.txt')
    train = base + get_requirements('train.txt')
    deploy = base + get_requirements('deploy.txt')
    return base, train, deploy

# 获取所有依赖
base_requirements, train_requirements, deploy_requirements = get_all_requirements()

setup(
    name="andy-llm",
    version="0.1.0",
    description="轻量级中文对话语言模型",
    author="Boss Andy",
    author_email="746144374@qq.com",
    url="https://github.com/DWG-ShowMaker/andy-llm",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require={
        'train': train_requirements,
        'deploy': deploy_requirements,
        'all': list(set(train_requirements + deploy_requirements))  # 去重
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
) 