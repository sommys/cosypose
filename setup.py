import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ['CXX'] = os.environ.get('GXX', '')

setup(
    ext_modules=[
        CppExtension(
            name='cosypose_cext',
            sources=['cosypose/csrc/cosypose_cext.cpp'],
            extra_compile_args=['-O3'],
            verbose=True
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
