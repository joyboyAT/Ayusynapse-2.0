from setuptools import setup, find_packages

setup(
    name="ayusynapse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch"
    ],
    python_requires=">=3.8",
)
