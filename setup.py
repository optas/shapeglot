from setuptools import setup

setup(name='shapeglot',
      version='0.1',
      description='ShapeGlot: Learning language for Shape Differentiation.',
      url='http://github.com/optas/shapeglot',
      author='Panos Achlioptas',
      author_email='optas@cs.stanford.edu',
      license='MIT',
      packages=['shapeglot'],
      install_requires=['pandas', 'torch', 'Pillow', 'numpy', 'matplotlib', 'six', 'nltk'],
      zip_safe=False)
