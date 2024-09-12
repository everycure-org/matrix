---
title: Installation
---

# Installation

This page assumes basic knowledge of the following technologies. We will provide instructions how to install them but as a quick reference, below are the necessary tools:

!!! info "Support on Windows, MacOS and Linux"
    We are mostly using MacOS but try our best to provide an onboarding for all
    platforms. This guide assumes Our guide assumes usage of [homebrew](https://brew.sh/)
    to manage packages on MacOS, [Windows
    WSL](https://de.wikipedia.org/wiki/Windows-Subsystem_f%C3%BCr_Linux) usage on Windows
    and some system proficiency for Linux users. If you find your platform could be
    better supported, do [send a
    PR](https://github.com/everycure-org/matrix/edit/main/src/docs/src/onboarding/index.md)!
    
### Python

We advise managing your Python installation using [`pyenv`](https://github.com/pyenv/pyenv).

=== "MacOS"

    ```bash
    brew install pyenv
    ```

### Python environment

We leverage [`uv`](https://github.com/astral-sh/uv) to manage/install our Python
requirements. Note that while many may be used to Conda, UV and Conda cannot be used in parallel. Using Conda is hence at your own risk.


Python 3.11 is currently **required** to build the matrix pipeline. If you attempt to use Python 3.12, you will likely encounter errors with the recently-removed `distutils` package (see the common errors document for how to solve this) 

Install as follows, then create a virtual env and install the requirements:


!!! warning
    Don't forget to link your uv installation using the instructions prompted after the downloaded.

=== "MacOS"

    ```bash
    brew install uv python@3.11
    ```

=== "Windows (WSL)"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Linux"
    ```bash
    # generic
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # for arch/manjaro
    sudo pacman -S uv
    ```

### Docker

Make sure you have [docker](https://www.docker.com/) and [docker-compose](https://docs.docker.com/compose/) installed. Docker can be downloaded directly from the from the [following page](https://docs.docker.com/get-docker/). 


=== "MacOS"

    ```bash
    brew install --cask docker #installs docker desktop
    brew install docker docker-compose #installs CLI commands
    ```

=== "Linux"

    ```bash
    sudo apt install docker #installs docker desktop
    brew install docker docker-compose #installs CLI commands
    ```


!!! Tip 
    The default settings of Docker have rather low resources configured, you might want to increase those in Docker desktop.

### Java

Our pipeline uses [Spark](https://spark.apache.org/) for distributed computations, which requires Java under the hood.

=== "MacOS"

    ```bash
    brew install openjdk@11
    brew link --overwrite openjdk@11 # makes the java version available in PATH
    ```

=== "Linux"

    ```bash
    # Java on Linux is complicated, check for your specific distro how to get JDK@11. 

    # On Arch/Manjaro
    pacman -S jdk11-openjdk
    ```

### gcloud SDK

We leverage Google (GCP) as our Cloud provider, the following cask installation provides CLI access to GCP.

=== "MacOS"

    ```bash
    brew install --cask google-cloud-sdk
    ```

After succesfully installation, authenticate the client:


```bash
gcloud auth login
gcloud auth application-default login
```

### GNU Make

We use `make` and `Makefile`s in a lot of places. If you want to [learn more about makefiles](https://makefiletutorial.com) feel free to do so. The essentials as a user are that you have it installed and can call it via CLI. 


=== "MacOS"

    ```bash
    # nothing to do here, make comes pre-installed with MacOS
    ```

=== "Windows (WSL)"

    ```bash
    sudo apt install build-essential
    ```

=== "Linux"
    ```bash
    # Debian based
    sudo apt install build-essential
    # for arch/manjaro
    sudo pacman -S make
    ```
