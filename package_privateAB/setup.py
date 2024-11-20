from setuptools import setup, find_packages

setup(
    name='privateAB',
    version='0.0.1',
    description='Two-sample testing (A/B testing) for multinomial and multivariate continuous data under local differential privacy',
    author='Jongmin Mun',
    author_email='jongmin.mun@marshall.usc.edu',
    url='https://jong-min-moon.github.io/softwares/',
    project_urls={
        "Bug Tracker": "https://github.com/Jong-Min-Moon/optimal-local-dp-two-sample",
    },
    install_requires = ['torch>=1.7.1', 'scipy>=1.7.3', 'numpy>=1.21.6', 'pandas>=1.3.5'],
    packages=find_packages(exclude=[]),
    keywords=['local differential privacy', 'A/B test', 'two-sample test', 'permutation test'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)