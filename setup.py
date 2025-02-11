from setuptools import setup, find_packages

setup(
    name="andy-llm",
    version="0.1.0",
    description="一个小型中文LLM模型训练-示例",
    author="[Boss Andy]",
    author_email="746144374@qq.com",
    url="https://github.com/DWG-ShowMaker/andy-llm",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "sentencepiece>=0.1.99",
        "wandb>=0.15.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 