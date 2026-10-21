from setuptools import setup, find_packages

setup(
    name="coreset",                # your package’s name
    version="0.1",                 # your version
    package_dir={"": "src"},       # tell setuptools where to find your code
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "ucimlrepo",
        "pandas",
    ],
)