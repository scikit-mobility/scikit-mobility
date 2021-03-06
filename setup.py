from setuptools import setup
from setuptools import find_packages

long_description = open('README.md').read()

REQUIRED_PKGS = ['python-igraph',
                 'pandas',
                 'geopandas',
                 'powerlaw',
                 'tqdm',
                 'requests',
                 'scikit-learn',
                 'statsmodels',
                 'folium',
                 'geojson',
                 'h3']
TESTS_REQUIRES = ['pytest']
EXTRAS_REQUIRE = {'test': TESTS_REQUIRES}
setup(
    name='scikit-mobility',
    packages=find_packages(include=["skmob", "skmob.*"]),
    version='1.2.2',
    extras_require=EXTRAS_REQUIRE,
    license='new BSD',
    python_requires='>=3.6',
    description='A toolbox for analyzing and processing mobility data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer='skmob Developers',
    maintainer_email='gianni.barlacchi@gmail.com',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 ],
    install_requires=REQUIRED_PKGS
    )
