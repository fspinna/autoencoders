from setuptools import setup

setup(
    name='autoencoders',
    version='0.0.1',
    packages=['autoencoders'],
    url='',
    license='',
    author='Francesco Spinnato',
    author_email='francesco.spinnato@sns.it',
    description='',
    install_requires=[
        'tensorflow>=2.3.0',
        'tensorflow-probability==0.12.2',
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'notebook',
        'scikit-learn'
    ]
)
