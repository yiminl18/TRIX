from setuptools import setup, find_packages

setup(
    name="trix",  # Replace with your library name
    version="0.1.0",  # Initial version
    description="A library that extracts structured data from templatized form-like documents automatically.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        "pytesseract",
        "pdfplumber",
        "pandas",
        "numpy",
        "scipy",
        "tiktoken",
        "pdf2image",
        "gurobipy",
        "openai",
        "Levenshtein",
        "boto3"
    ],
)
