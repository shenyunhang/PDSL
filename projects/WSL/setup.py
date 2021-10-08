#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "wsl", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("D2_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "wsl", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )

    hipify_ver = (
        [int(x) for x in torch.utils.hipify.__version__.split(".")]
        if hasattr(torch.utils.hipify, "__version__")
        else [0, 0, 0]
    )

    if is_rocm_pytorch and hipify_ver < [1, 0, 0]:  # TODO not needed since pt1.8

        # Earlier versions of hipification and extension modules were not
        # transparent, i.e. would require an explicit call to hipify, and the
        # hipification would introduce "hip" subdirectories, possibly changing
        # the relationship between source and header files.
        # This path is maintained for backwards compatibility.

        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/detectron2/layers/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        shutil.copy(
            "detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h",
            "detectron2/layers/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h",
        )
        shutil.copy(
            "detectron2/layers/csrc/deformable/deform_conv.h",
            "detectron2/layers/csrc/deformable/hip/deform_conv.h",
        )

        sources = [main_source] + sources
        sources = [
            s
            for s in sources
            if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
        ]

    else:

        # common code between cuda and rocm platforms,
        # for hipify version [1,0,0] and later.

        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

        sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    # DenseCRF
    crf_sources = glob.glob(os.path.join(extensions_dir, "crf", "densecrf", "src", "*.cpp"))
    crf_include_dirs = os.path.join(extensions_dir, "crf", "densecrf", "include")
    for source in crf_sources:
        if "optimization" in source:
            continue
        if "examples" in source:
            continue
        sources += [source]
    include_dirs += [crf_include_dirs]

    ext_modules = [
        extension(
            "wsl._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# For projects that are relative small and provide features that are very close
# to detectron2's core functionalities, we install them under detectron2.projects
PROJECTS = {}

setup(
    name="wsl",
    version=get_version(),
    author="Yunhang Shen",
    url="https://github.com/shenyunhang",
    packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    python_requires=">=3.6",
    install_requires=["graphviz"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
