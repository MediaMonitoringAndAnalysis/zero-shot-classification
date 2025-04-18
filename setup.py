from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zero-shot-classification",
    version="0.1.0",
    author="MMP Team",
    author_email="your.email@example.com",
    description="A two-stage zero-shot classifier that uses a smaller model for initial filtering and a larger LLM for more precise classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zero-shot-classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "nltk>=3.8.1",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "llm-multiprocessing-inference>=0.1.0",
    ],
) 