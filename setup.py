from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="spex",
    version="0.1.0",
    author="talargv",
    author_email="talargov1@mail.tau.ac.il",
    description="SpEx: A Spectral Approach to Explainable Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talargv/spex",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)