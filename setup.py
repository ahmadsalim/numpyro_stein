import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numpyro_stein", # Replace with your own username
    version="0.0.1",
    author="Ahmad Salim Al-Sibahi",
    description="Stein Variational Inference for NumPyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmadsalim/numpyro_stein",
    packages=setuptools.find_packages(include=['numpyro_stein', 'numpyro_stein.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpyro", "matplotlib", "seaborn",
                      "jax>=0.1.63", "jaxlib>=0.1.43"],
    python_requires='>=3.6',
)