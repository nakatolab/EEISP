import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eeisp",
#    version="0.6.57", # for test
    version="0.6.0", # for PyPI
    license="GPL3.0",
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.22.0",
        "scipy>=1.3",
        "matplotlib",
        "seaborn",
        "networkx",
        "igraph"
    ],
    author="Ryuichiro Nakato",
    author_email="rnakato@iqb.u-tokyo.ac.jp",
    description="Identify gene pairs that are codependent and mutually exclusive from single-cell RNA-seq data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nakatolab/EEISP",
    keywords="eeisp scRNA-seq",
    scripts=['eeisp/eeisp',
             'eeisp/eeisp.heatmap',
             'eeisp/eeisp.Louvain',
             'eeisp/eeisp.LouvainSigned',
             'eeisp/eeisp_add_genename_from_geneid'
             ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
