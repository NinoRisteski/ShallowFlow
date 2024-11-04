from setuptools import setup, find_packages

setup(
    name="shallowflow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets",
        "accelerate"
    ],
)