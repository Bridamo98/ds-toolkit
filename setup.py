from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ds_toolkit",
    packages=find_packages(include=["ds_toolkit", "ds_toolkit.*"]),
    version="0.1.0",
    description="Python library for Data Science",
    author="Briam Agudelo",
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest==8.0.2"],
    test_suite="tests",
)
