from setuptools import setup, find_packages

def get_requirements(filename):
    with open(f'requirements/{filename}') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="andy-llm",
    version="0.1.0",
    description="一个小型中文LLM模型训练-示例",
    author="[Boss Andy]",
    author_email="746144374@qq.com",
    url="https://github.com/DWG-ShowMaker/andy-llm",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=get_requirements('deploy.txt'),
    extras_require={
        'train': get_requirements('train.txt'),
        'all': get_requirements('train.txt') + get_requirements('deploy.txt')
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 