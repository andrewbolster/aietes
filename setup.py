import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
	"SimPy>=2.3",
    "numpy>=1.6",
    "scipy>=0.9",
    "pydot>=1.0",
    "wxPython>=2.8", # Use the python-wxgtk2.8 package
    "matplotlib>=1.1",
    "pykalman"
]


setup(name='aietes',
    version=version,
    description="A Multi-Node behaviour simulator for AUV Comms Research",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='AUV Behaviour Research Python simulation simpy numpy communications',
    author='Andrew Bolster',
    author_email='me@andrewbolster.info',
    url='http://andrewbolster.info',
    license='',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    package_data = {
        'aietes' : ['configs/*.conf'],
        'ephyra' : ['icons/*.png']
    },
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['aietes=aietes:main',
             'bounos=bounos:main',
             'ephyra=ephyra:main'
            ]
    }
)
