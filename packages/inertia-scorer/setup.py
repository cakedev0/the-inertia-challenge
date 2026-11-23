from setuptools import Extension, setup

# setuptools compiles the .pyx with the Cython of the build requirements.
setup(ext_modules=[Extension("inertia_scorer._sums", ["src/inertia_scorer/_sums.pyx"])])
