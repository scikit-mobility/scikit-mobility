from setuptools import setup
import sys
if sys.version_info >= (3,8):
    sys.exit('Sorry, Python > 3.7 is not supported')
from setuptools import setup, find_packages

long_description = open('README.md').read()

DEPENDENCIES = ['numpy==1.17.0', 'pandas==0.24', 'geopandas==0.5.0', 'scipy==1.3.0', 'powerlaw==1.4.4',
                'tqdm==4.32.1', 'requests==2.22.0', 'scikit-learn==0.21.2', 'statsmodels==0.10.0rc2',
                'folium==0.9.1', 'matplotlib==3.1.1', 'geojson==2.4.1', 'shapely==1.7a1', 'fiona==1.8.6']

TEST_DEPENDENCIES = [
    'hypothesis',
    'mock',
    'python-Levenshtein',
]


setup(
    name='scikit-mobility',
    version='1.0',
    packages=['skmob', 'skmob.core', 'skmob.utils', 'skmob.io', 'skmob.measures', 'skmob.models', 'skmob.preprocessing', 'skmob.privacy', 'skmob.tessellation' ],
    #TODO: fix it with find_packages(include=["skmob", "skmob.*"])
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
