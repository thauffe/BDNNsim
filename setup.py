from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="BDNNsim",
    version="0.1",
    packages=find_packages(),
    py_modules=["runner"],  # Include the runner script as a module
    entry_points={
        "console_scripts": [
            "BDNNsim = runner:main",  # Command: `JaxRate`
        ],
    },
    description="Birth-death simulation of diversification and fossil sampling",
    author="Torsten Hauffe",
    author_email="torsten.hauffe@ikmail.com",
    python_requires=">=3.11",
    install_requires=read_requirements()
)
