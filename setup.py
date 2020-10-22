import os
from setuptools import setup
from collections import defaultdict

__version__ = 1.0
assert __version__

install_requires = [
    'SpiNNUtilities >= 1!5.0.1, < 1!6.0.0',
    'SpiNNMachine >= 1!5.0.1, < 1!6.0.0',
    'SpiNNMan >= 1!5.0.1, < 1!6.0.0',
    'SpiNNaker_PACMAN >= 1!5.0.1, < 1!6.0.0',
    'SpiNNaker_DataSpecification >= 1!5.0.1, < 1!6.0.0',
    'spalloc >= 2.0.2, < 3.0.0',
    'SpiNNFrontEndCommon >= 1!5.0.1, < 1!6.0.0',
    'numpy', 'lxml', 'six']
if os.environ.get('READTHEDOCS', None) != 'True':

    # scipy must be added in config.py as a mock
    install_requires.append('scipy')


# Build a list of all project modules, as well as supplementary files
main_package = "spinn_gym"
extensions = {".aplx", ".boot", ".cfg", ".json", ".sql", ".template", ".xml",
              ".xsd", ".dict"}
main_package_dir = os.path.join(os.path.dirname(__file__), main_package)
start = len(main_package_dir)
packages = []
package_data = defaultdict(list)
for dirname, dirnames, filenames in os.walk(main_package_dir):
    if '__init__.py' in filenames:
        package = "{}{}".format(
            main_package, dirname[start:].replace(os.sep, '.'))
        packages.append(package)
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        if ext in extensions:
            package = "{}{}".format(
                main_package, dirname[start:].replace(os.sep, '.'))
            package_data[package].append(filename)

setup(
    name="SpiNNGym",
    version=__version__,
    description="SpiNNaker Gym for reinforcement learning with SNNs",
    url="https://github.com/SpiNNakerManchester/SpiNNGym",
    packages=packages,
    package_data=package_data,
    install_requires=install_requires
)