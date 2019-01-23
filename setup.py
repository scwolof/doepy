
from setuptools import setup, find_packages
from pathlib import Path

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'GPdode' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

setup(
    name='GPdode',
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    version=about['__version__'],
    url='https://github.com/scwolof/GPdode',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.7', 'GPy'],
    setup_requires=['numpy>=1.7', 'GPy'],
    tests_require=['pytest-runner','pytest', 'pytest-cov'],
)
