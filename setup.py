import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openmc_plasma_source",
    version="develop",
    author="The openmc-plasma-source Development Team",
    author_email="rdelaportemathurin@gmail.com",
    description="Creates tokamak plasma sources for OpenMC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fusion-energy/openmc-plasma-source",
    packages=setuptools.find_packages(),
    classifiers=[
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    tests_require=[
        "pytest",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.9',
    ],
)
