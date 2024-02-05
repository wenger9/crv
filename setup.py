from setuptools import find_packages, setup
from gda_clv import __version__

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="gda_clv",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    authors="",
    include_package_data=True,
    install_requires=required,
)
