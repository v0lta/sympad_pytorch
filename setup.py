from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="sympad",
    version="0.0.1",
    ext_modules=[cpp_extension.CppExtension("sympad", ["src/sympad.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    install_requires=["torch"]
)
