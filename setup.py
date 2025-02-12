import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eeisp",
    version="0.6.1", # for test
#    version="0.5.0",
    license="GPL3.0",
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.22.0",
        "scipy>=1.3",
        "networkx"
    ],
    author="Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="Identify gene pairs that are codependent and mutually exclusive from single-cell RNA-seq data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nakatolab/EEISP",
    keywords="eeisp scRNA-seq",
    scripts=['eeisp/eeisp', 'eeisp/eeisp_add_genename_from_geneid'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
