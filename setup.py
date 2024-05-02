import os
import setuptools
import shutil


long_desc = """# spRAG

State-of-the-art RAG pipeline from Superpowered AI.
"""


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r", encoding="utf-8") as fh:
        return fh.read()


setuptools.setup(
    name="sprag",
    version=read("VERSION"),
    description="spRAG",
    license="MIT License",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/SuperpoweredAI/spRAG",
    project_urls={
        "Homepage": "https://github.com/SuperpoweredAI/spRAG",
        "Documentation": "https://github.com/SuperpoweredAI/spRAG",
        "Contact": "https://github.com/SuperpoweredAI/spRAG",
    },
    author="Superpowered AI",
    author_email="zach@superpowered.ai, justin@superpowered.ai",
    packages=["sprag"],
    install_requires=read("requirements.txt"),
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
