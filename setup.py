from setuptools import setup, find_packages

setup(
    name='autoencoders',
    version='0.0.1',
    packages=['autoencoders'],  # find_packages(),  # ['autoencoders'],
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
        'scikit-learn',
        'setuptools-git'
    ],
    package_data={'': ['*.npy', '*.h5', '*.joblib']},
    include_package_data=True
)
