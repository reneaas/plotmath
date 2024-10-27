import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plotmath",  # Replace with your desired package name
    version="0.1.9",
    author="René Alexander Ask",
    author_email="rene.ask@icloud.com",
    description="Automatically generates textbook graphs for mathematical functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reneaas/plotmath",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
