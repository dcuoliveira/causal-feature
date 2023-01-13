import setuptools #python3 -m pip install setuptools

#4. Now you can add code files inside the quantlib folder and use it as a package like any other! (pip install using developement mode) >> python3 -m pip install -e.

setuptools.setup(
    name="causal_feature",
    version="0.1",
    description="",
    url="#",
    author="dcuoliveira",
    # install_requires=["opencv-python"],
    # author_email="",
    packages=setuptools.find_packages(),
    zip_safe=False
)