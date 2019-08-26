from setuptools import setup, find_packages

long_description = open('README.md').read()

DEPENDENCIES = ['numpy', 'pandas==0.24', 'geopandas', 'scipy', 'powerlaw', 'tqdm', 'osmnx', 'scikit-learn',
                'statsmodels', 'folium', 'matplotlib', 'geojson', 'shapely', 'geopy', 'fiona']

TEST_DEPENDENCIES = [
    'hypothesis',
    'mock',
    'python-Levenshtein',
]


setup(
    name='scikit-mobility',
    version='1.0',
    packages=['skmob', 'skmob.core', 'skmob.utils', 'skmob.io', 'skmob.measures', 'skmob.models', 'skmob.preprocessing', 'skmob.privacy', 'skmob.tessellation' ]
    #TODO: fix it with find_packages(include=["skmob", "skmob.*"]),
    license='MIT',
    python_requires='>=3.6',
    description='A toolbox for analyzing and processing mobility data.',
    long_description=long_description,
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
                 #'Programming Language :: Python :: 3',
                 #'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 ],
    install_requires=DEPENDENCIES
    #extra_requires={'geometry_support':'geopandas'}
    )
