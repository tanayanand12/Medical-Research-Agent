from setuptools import setup, find_packages

setup(
    name="fda-agent",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.21.0",
        "tenacity>=8.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "fastapi>=0.68.0",
        "pydantic>=1.8.0",
        "uvicorn>=0.15.0",
        "backoff>=2.0.0"
    ],
    author="Tanay Anand",
    author_email="tanay.anand12personal@gmail.com",
    description="A RAG-based FDA data analysis library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fda-agent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
