#!/usr/bin/python3
# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""XLA build script for use in CI."""
import dataclasses
import enum
import functools
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}

# TODO(ddunleavy): move this to the bazelrc
_DEFAULT_OPTIONS = dict(
    test_output="errors",
    keep_going=True,
    nobuild_tests_only=True,
    features="layering_check",
    profile="/tf/pkg/profile.json.gz",
    flaky_test_attempts=3,
    jobs=150,
)

_DEFAULT_TARGET_PATTERNS = ("//xla/...", "//build_tools/...", "@local_tsl//tsl/...")

_COMPUTE_CAPABILITIES = (60, 70, 80, 90)


class BuildType(enum.Enum):
  CPU_X86 = enum.auto()
  CPU_ARM64 = enum.auto()
  GPU = enum.auto()
  GPU_CONTINUOUS = enum.auto()


@dataclasses.dataclass(**_KW_ONLY_IF_PYTHON310)
class Build:
  """Class representing a build of XLA."""

  type_: dataclasses.InitVar[BuildType]
  configs: tuple[str, ...] = ()
  target_patterns: tuple[str, ...] = _DEFAULT_TARGET_PATTERNS
  tag_filters: tuple[str, ...] = ()
  options: dict[str, Any] = dataclasses.field(default_factory=dict)
  compute_capability: dataclasses.InitVar[int | None] = None

  def __post_init__(self, type_: BuildType, compute_capability: int | None):
    if type_ in (BuildType.GPU, BuildType.GPU_CONTINUOUS):
      assert (
          compute_capability
      ), "Compute capability must be given for GPU builds"
      self.tag_filters += (f"requires-gpu-sm{compute_capability}-only",)
      for cc in _COMPUTE_CAPABILITIES:
        if compute_capability >= cc:
          self.tag_filters += (f"requires-gpu-sm{cc}",)

  def bazel_test_command(self) -> list[str]:
    # pylint: disable=g-bool-id-comparison
    options = [
        f"--{k}" if v is True else f"--{k}={v}" for k, v in self.options.items()
    ]
    configs = [f"--config={config}" for config in self.configs]
    build_tag_filters = f"--build_tag_filters={','.join(self.tag_filters)}"
    test_tag_filters = f"--test_tag_filters={','.join(self.tag_filters)}"
    all_options = configs + options + [build_tag_filters, test_tag_filters]
    return ["bazel", "test", *all_options, "--", *self.target_patterns]


nvidia_gpu_build_with_compute_capability = functools.partial(
    Build,
    configs=("warnings", "rbe_linux_cuda_nvcc"),
    tag_filters=("-no_oss", "requires-gpu-nvidia"),
    options=dict(
        run_under="//tools/ci_build/gpu_build:parallel_gpu_execute",
        **_DEFAULT_OPTIONS,
    ),
)

_CPU_X86_BUILD = Build(
    type_=BuildType.CPU_X86,
    configs=("-no_oss", "nonccl", "rbe_linux_cpu"),
    target_patterns=_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=("-no_oss", "-gpu", "-requires-gpu-nvidia"),
    options=_DEFAULT_OPTIONS,
)
_CPU_ARM64_BUILD = Build(
    type_=BuildType.CPU_ARM64,
    configs=("nonccl", "rbe_cross_compile_linux_arm64_xla"),
    target_patterns=_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=("-no_oss", "-gpu", "-requires-gpu-nvidia"),
    options=_DEFAULT_OPTIONS,
)
_GPU_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU, compute_capability=75
)
_GPU_CONTINUOUS_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU_CONTINUOUS, compute_capability=80
)

_KOKORO_JOB_NAME_TO_BUILD_MAP = {
    "tensorflow/xla/linux/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/linux/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/build_gpu": _GPU_CONTINUOUS_BUILD,
    "tensorflow/xla/linux/github_continuous/build_cpu": _CPU_X86_BUILD,
}


def _write_to_sponge_config(key, value) -> None:
  artifacts_dir = pathlib.Path(os.getenv("KOKORO_ARTIFACTS_DIR"))
  with (artifacts_dir / "custom_sponge_config.csv").open("a") as f:
    f.write(f"{key},{value}\n")


def _logged_subprocess(args, **kwargs):
  logging.info("Starting process: %s", " ".join(args))
  # pylint: disable=subprocess-run-check
  return subprocess.run(args, **kwargs)


def _pull_docker_image_with_retries(image_url: str, retries=3) -> None:
  """Pulls docker image with retries to avoid transient rate limit errors."""
  for _ in range(retries):
    pull_proc = _logged_subprocess(["docker", "pull", image_url], check=False)
    if pull_proc.returncode != 0:
      time.sleep(15)

  # write SHA of image to the sponge config
  _write_to_sponge_config("TF_INFO_DOCKER_IMAGE", image_url)
  sha_proc = _logged_subprocess(
      [
          "docker",
          "inspect",
          "--format'{{index .RepoDigests 0}}'",
          image_url,
      ],
      capture_output=True,
      check=True,
  )
  _, sha = str(sha_proc.stdout).split("@")
  _write_to_sponge_config("TF_INFO_DOCKER_SHA", sha)


def main():
  docker_image = os.getenv("DOCKER_IMAGE")
  kokoro_job_name = os.getenv("KOKORO_JOB_NAME")
  kokoro_artifacts_dir = os.getenv("KOKORO_ARTIFACTS_DIR")
  build = _KOKORO_JOB_NAME_TO_BUILD_MAP[kokoro_job_name]

  _logged_subprocess(
      [
          f"{kokoro_artifacts_dir}/github/xla/.kokoro/generate_index_html.sh",
          f"{kokoro_artifacts_dir}/index.html",
      ],
      check=True,
  )

  _pull_docker_image_with_retries(docker_image)

  _logged_subprocess(
      # pyformat: disable
      [
          "docker", "run",
          "--name", "xla",
          "-w", "/tf/xla",
          "-itd",
          "--rm",
          "-v", f"{kokoro_artifacts_dir}/github/xla:tf/xla",
          "-v", f"{kokoro_artifacts_dir}/pkg:tf/pkg",
          docker_image,
          "bash",
      ],
      # pyformat: enable
      check=True,
  )

  _logged_subprocess(
      ["docker", "exec", "xla", *build.bazel_test_command()], check=True
  )

  _logged_subprocess([
      "docker",
      "exec",
      "xla",
      "bazel",
      "analyze-profile",
      "/tf/pkg/profile.json.gz",
  ])

  _logged_subprocess(["docker", "stop", "xla"])


if __name__ == "__main__":
  main()
