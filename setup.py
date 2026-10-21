from setuptools import setup, find_packages

setup(
    name="grid-coresets",
    version="0.1",
    description="Grid-based coreset construction for accelerating k-means clustering",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "ucimlrepo",
    ],
)
