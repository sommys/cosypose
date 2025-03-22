from os import path, environ
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

environ["CXX"] = environ.get("GXX", "g++")

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name="cosypose_cext",
            sources=["cosypose/csrc/cosypose_cext.cpp"],
            extra_compile_args=["-O3"],
            verbose=True,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    print("Warning: torch is not installed in the installer venv. Skipping C++ extension build.")
    ext_modules = []
    cmdclass = {}

setup(
    name="cosypose",
    version="1.0.0",
    description="CosyPose",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
