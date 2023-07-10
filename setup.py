# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import multiprocessing
import os
import re
import subprocess
import sys
import sysconfig

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test

class CMakeExtension(Extension):
    def __init__(self, name, src_dir=""):
        super(CMakeExtension, self).__init__(name, sources=[])
        self.src_dir = os.path.abspath(src_dir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            cmake_version = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", cmake_version.decode()).group(1))
        if cmake_version < "3.14":
            raise RuntimeError("CMake >= 3.14 is required.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" +
                      ext_dir, "-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", f"-j{multiprocessing.cpu_count()}"]

        env = os.environ.copy()
        env["CXXFLAGS"] = f'{env.get("CXXFLAGS", "")} \
                -DVERSION_INFO="{self.distribution.get_version()}"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.src_dir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] +
                              build_args, cwd=self.build_temp)

        print()  # Add an empty line for cleaner output


setup(
    name="ml4co",
    version="0.1",
    packages=["ml4co", "ml4co.ops"],
    description="",
    long_description="",
    # add extension module
    ext_modules=[CMakeExtension("ml4co", "./ml4co")],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    url="",
)
