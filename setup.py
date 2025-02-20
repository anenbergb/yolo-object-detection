from setuptools import setup, find_packages

setup(
    name="yolo",
    version="1.0.0",
    url="https://github.com/anenbergb/MBAS",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="A fully from scratch implementation of a Yolo style object detector",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorboard",
        "pandas",
        "pandas-stubs",
        "opencv-python",
        "loguru",
        "matplotlib",
        "ffmpeg-python",
        "tqdm",
        "types-tqdm",
        "pillow",
        "types-Pillow",
        "tabulate",
        "fiftyone",
        "pycocotools",
        "torchmetrics[detection]",
    ],
    extras_require={
        "torch": [
            "torch",
            "torchvision",
        ],
        "notebook": [
            "jupyter",
            "itkwidgets",
            "jupyter_contrib_nbextensions",
            "plotly",
            "seaborn",
        ],
        "dev": ["black", "mypy", "flake8", "isort", "ipdb"],
    },
)
