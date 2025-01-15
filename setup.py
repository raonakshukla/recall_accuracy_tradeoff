import setuptools

version = '0.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='recall_optimisation',  
     version=version,
     author="Raonak Shukla and Rong Qu",
     author_email="raonakshukla@gmail.com and pszrq@exmail.nottingham.ac.uk",
     description="A Python Package for Optimising Recall for Binary Classification",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="to_be _entered",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )
