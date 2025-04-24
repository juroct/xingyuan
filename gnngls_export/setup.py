from setuptools import setup, find_packages

setup(
    name="gnngls",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "dgl>=1.1.0",
        "networkx>=3.2.0",
        "numpy<2.0",
        "torchdata",
        "matplotlib",
        "pandas",
        "tqdm",
        "scikit-learn",
        "tensorboard",
        "concorde-tsp",
        "python-lkh",
        "tsplib95",
    ],
    python_requires=">=3.10",
)
