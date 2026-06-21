from setuptools import setup, find_packages

setup(
    name="earthquake-forecasting-central-asia",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.2",       # include_groups=False requires 2.2+
        "scipy>=1.10",
        "geopandas>=0.14",
        "shapely>=2.0",
        "requests>=2.31",
        "torch>=2.1",
        "lightgbm>=4.1",
        "catboost>=1.2",
        "scikit-learn>=1.3",
        "pycsep>=0.6",
        "contextily>=1.4",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
    ],
)
