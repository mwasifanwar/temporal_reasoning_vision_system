from setuptools import setup, find_packages

setup(
    name="temporal-reasoning-vision-system",
    version="1.0.0",
    author="mwasifanwar",
    description="Advanced computer vision system for temporal reasoning, causal analysis, and event prediction in video sequences",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "pytest>=7.3.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.2.0"
    ],
    python_requires=">=3.8",
)