"""Setup script for Intain Quant package."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intain-quant",
    version="0.1.0",
    author="IDA Team",
    author_email="analytics@intainft.com",
    description="Monte Carlo collateral engine for residential mortgage portfolios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intainft/intain-quant",
    packages=find_packages(include=[
        "core", "core.*",
        "data_prep", "data_prep.*",
        "distributions", "distributions.*",
        "behaviors", "behaviors.*",
        "engine", "engine.*",
        "pm", "pm.*",
        "models", "models.*",
        "app", "app.*",
    ]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "streamlit>=1.28.0",
        "plotly>=5.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "python-dateutil>=2.8.2",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
            "ipykernel>=6.23.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "ml": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "intain-quant=app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
