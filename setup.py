import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npsn",
    version="0.1",
    author="Akshay J. Dave",
    author_email="akshayjd@mit.edu",
    description="Nuclear Power Surrogate Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a-jd/npsn",
    packages=setuptools.find_packages(),
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow==2.2.0',
        'hyperopt==0.2.4',
        'scikit-learn==0.23.1',
        'pydoe==0.3.8',
        'tables==3.6.1'
    ],
    zip_safe=False,
)
