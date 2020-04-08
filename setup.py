import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numpyro-stein", # Replace with your own username
    version="0.0.1",
    author="Ahmad Salim Al-Sibahi",
    description="Stein Variational Inference for NumPyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmadsalim/numpyro-stein",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpyro", "matplotlib", "seaborn"],
    python_requires='>=3.6',
)