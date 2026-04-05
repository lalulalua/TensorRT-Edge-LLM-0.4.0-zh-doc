> **Modified:** This file was changed after the NVIDIA TensorRT-Edge-LLM 0.4.0 release for this unofficial distribution. See [PROJECT_ORIGIN.md](PROJECT_ORIGIN.md).

# TensorRT Edge-LLM Contribution Rules

## Issue Tracking

* **Upstream NVIDIA TensorRT-Edge-LLM:** For defects and feature requests that should land in NVIDIA’s official project, use the [upstream issue tracker](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues) and follow NVIDIA’s contribution process there.
* **This repository (unofficial distribution):** For problems or changes specific to **this** fork or redistribution (for example, `docs_zh/`, packaging, or notices in this repo), open an issue in **this** repository’s issue tracker on the Git host where you cloned or forked it.

## Coding Guidelines

* Coding style for TensorRT-Edge-LLM can be found [in this document](CODING_GUIDELINES.md).

* All contributed C++ code should be formatted following the rules in TensorRT Edge-LLM's [clang-format](.clang-format) file. The recommended version is clang-format>=14.0.

* Changes can be formatted with the following command:

  ```bash
  # Commit ID is optional - if unspecified, run format on staged changes.
  git-clang-format --style file [commit ID/reference]
  ```

* All contributed Python code should be formatted using the `black` Python package. The recommended version is `black>=23.0`

* Changes can be formatted with the following command:

  ```bash
  git diff --name-only | grep "*.py" | xargs black -l 120
  ```

* Try to keep pull requests (PRs) as concise as possible:
  * Avoid committing commented-out code.
  * Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

## Coding Style

We use `pre-commit` for automatic code formatting and validation. Install the `pre-commit` package in your local
Python environment.

```bash
pip install pre-commit
pre-commit install
```

`pre-commit` will be triggered in every commit.

```bash
git commit -m "fix"

isort....................................................................Passed
CRLF end-lines remover...................................................Passed
yapf.....................................................................Failed
- hook id: yapf
- files were modified by this hook
check for added large files..............................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
detect private key.......................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
trim trailing whitespace.................................................Passed
check toml...............................................................Passed
mixed line ending........................................................Passed
debug statements (python)................................................Passed
check json...........................................(no files to check)Skipped
autoflake................................................................Passed
clang-format.............................................................Passed
cmake-format.............................................................Passed
codespell................................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
mdformat.................................................................Passed
```

If any files were modified by this hook, you will need to stage and commit them again.


## Pull Requests

### Developer workflow
Developer workflow for code contributions to **this** repository is as follows:

1. [Fork](https://help.github.com/en/articles/fork-a-repo) **this** repository on your Git host (or clone it with write access if you are a collaborator). To contribute to NVIDIA’s **upstream** project instead, use [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).

2. Clone the forked repository and push changes to your fork.

  ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git TensorRT-Edge-LLM
    # Checkout the targeted branch and commit changes
    # Push the commits to a branch on the fork (remote).
    git push -u origin <local-branch>:<remote-branch>
  ```

3. Once the code changes are staged on the fork and ready for review, open a [Pull Request](https://help.github.com/en/articles/about-pull-requests) to merge your branch into the target branch of **this** repository (often `main` or `master`, depending on how this repo is configured).
  * Creation of a PR kicks off the review process for **this** repo.
  * Maintainers of **this** repository will review PRs when available. Labels such as `Pending Review` or `Changes Requested` may be used similarly to common open-source practice.
  * If CI is configured on this repository, fix reported failures before requesting another review.

### PR Submission Policies

The naming of merge requests in TensorRT-Edge-LLM follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). If the PR includes an API change that might break user code/API usage, consider adding "BREAKING CHANGE" in the title so that reviewers know what to expect. Additionally, if the PR is not related to any bug and task, consider using "chore" or None as the placeholder.

[!IMPORTANT]
For contributions intended for **NVIDIA’s upstream** repository: NVIDIA developers should include the JIRA number or NVBUG ID in the PR title when required by NVIDIA’s process, and follow internal branch rules described in the upstream project.

Good PR Titles Examples:
* feat: Add support for starcoder-v2 FP8 base + FP16/BF16 LoRA
* BREAKING CHANGE: Set default max batch size to 2048
* chore: Remove version from plugins .so
* None: Stringized enums for better error msgs
* fix https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/700: a Memory leak issue in C++ runtime


This is important for tracking and collecting what has been submitted to which release and makes it easier for others to track the bugs or tasks. It could also be helpful when collecting GitHub publish announcement.

In the PR description, please consider addressing these points:

* Background or motivation for the PR: Why is the change necessary?
* Summarize the changes in one paragraph, if possible.
* If the PR is large, explain why it cannot be broken down into multiple PRs.
* Potential performance or functional impacts of the changes. If there are risks, please inform the reviewers.
* Link to the related PRs.

[!IMPORTANT]
For **NVIDIA upstream** contributions: submit feature or bug fixes to the branch specified in NVIDIA’s bug-tracking workflow (for example `release/0.4.0` when applicable). Add a "release blocker" label only when using NVIDIA’s upstream tracker and labels.


## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license. Signing off your commit means you accept the terms of the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
Welcome to contribute to TensorRT-Edge-LLM :)
