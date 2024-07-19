from setuptools import find_packages, setup

setup(
    name="frequentist_matching_priors",
    packages=find_packages(where="src"),
    package_dir={"" : "src"},
    requires=[
        "numpy",
        "matplotlib",
        "jax[cpu]",
        "blackjax",
        "tqdm",
        "pandas",
        "chainconsumer",
        "tensorflow_probability"
    ]
)