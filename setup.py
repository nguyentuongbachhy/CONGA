"""
Setup script for CONGA package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="conga",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CONGA: COntrastive Nested Graph Architecture for Continual Sequential Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/conga",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "conga-train=scripts.train:main",
            "conga-eval=scripts.evaluate:main",
            "conga-benchmark=scripts.benchmark:main",
        ],
    },
)
