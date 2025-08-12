---
title: Installation
---

# Installation

In this page we are installing tools related to the technology stack which we introduced [earlier](./tech_stack.md):

!!! info "Support on Windows, MacOS and Linux"
    We are mostly using MacOS but try our best to provide an onboarding for all
    platforms. This guide assumes usage of [homebrew](https://brew.sh/)
    to manage packages on MacOS, [Windows
    WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) usage on Windows
    and some system proficiency for Linux users. If you find your platform could be
    better supported, do [send a
    PR](https://github.com/everycure-org/matrix/edit/main/src/docs/src/onboarding/index.md)!

??? note "Installing Windows Subsystem for Linux (WSL)"

    If you are running on Windows, you need to install Windows Subsystem for Linux as the following steps require a UNIX OS. You can follow this [tutorial from Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install). 

    === "Windows (Powershell)"
    
        ```bash
        wsl --install
        ```

    If using WSL, you need to ensure the MATRIX Github repo is cloned within WSL.

    ??? Tip "Cloning Github repos in WSL"

        Cloning repos by HTTPS within WSL is no longer supported and using SSH key is recommended. You can set it up by following the Github tutorials on [generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux) and [adding a new SSH key to your account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

        === "WSL"

            ```bash
            # generate a new SSH key, using your Github login email address
            ssh-keygen -t ed25519 -C "your_email@example.com"
            # then you need to enter a passphrase
            # add ssh-key to your ssh agent
            # start ssh-agent in the background
            eval "$(ssh-agent -s)"
            # add ssh private key to the ssh-agent
            ssh-add ~/.ssh/id_ed25519

            # add a new ssh key to your account
            cat ~/.ssh/id_ed25519.pub
            # Then select and copy the contents of the id_ed25519.pub file
            # displayed in the terminal to your clipboard
            # then following steps 2-9 on the Github tutorial on adding a new SSH key to your account listed above
            ```

### Python
We advise managing your Python installation using [`pyenv`](https://github.com/pyenv/pyenv).

=== "MacOS"

    ```bash
    brew install pyenv
    ```

=== "Windows (WSL)"

    Steps for installing pyenv in WSL following the [tutorial on Github](https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv). First, install dependencies (if not already installed):

    ```bash
    sudo apt-get update; sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

    Then clone the pyenv repository:

    ```bash
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    ```

    Define the PYENV_ROOT environment variable:

    ```bash
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    ```

    Enable pyenv init:

    ```bash
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
    ```

    Restart your shell so the changes take effect:

    ```bash
    exec "$SHELL"
    ```

    *Check the pyenv version*:

    ```bash
    pyenv --version
    ```

    This should print the version of pyenv that you have installed, for example: ```bash pyenv 2.3.6.```

    After following these steps, you should have pyenv installed and ready to use on your WSL environment.


Once `pyenv` is installed, you can install the latest version of Python 3.11 using the command:

```bash
pyenv install 3.11
```

After `pyenv` installs Python, you can check that your global Python version is indeed 3.11:

```bash
pyenv global
# should print 3.11
```

You can also try running `python` from the command line to check that your global Python version is indeed some version of 3.11 (3.11.11 is the latest version of Python 3.11 as of December 12, 2024).

```bash
python
# the first line printed by the Python interpreter should say something like
# Python 3.11.11 (main, Dec 12 2024, 13:48:23) [Clang 16.0.0 (clang-1600.0.26.6)]
# The exact details of the message might differ---the main thing is that you are running
# Python 3.11.<something>, as opposed to another version of Python, such as 3.9, 3.12, or 3.13.
```

### uv installation

We leverage [`uv`](https://github.com/astral-sh/uv) to manage our Python dependencies and workspace structure. uv provides fast, reliable package management and replaces the need for pip, requirements.txt files, and virtual environment management.

!!! warning "Conda Incompatibility"
    uv and Conda cannot be used together. If you have Conda installed, you may encounter conflicts. We strongly recommend using uv exclusively for this project.

Python 3.11 is currently **required** to build the matrix pipeline. If you attempt to use Python 3.12, you will likely encounter errors with the recently-removed `distutils` package (see the common errors document for how to solve this) 

!!! warning
    Don't forget to link your uv installation using the instructions prompted after the downloaded.

=== "MacOS"
    If you have installed Python 3.11 using `pyenv`, as recommended above, you just need to install `uv`:

    ```bash
    brew install uv
    ```

    If, however, you prefer to install Python 3.11 using Homebrew, you need to install both `uv` and Python:

    ```bash
    brew install uv python@3.11
    ```


=== "Windows (WSL)"

    ```bash
    # install uv
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

=== "Windows (WSL)"
    
    ```bash
    # install docker
    sudo apt install docker
    # install docker-compose
    sudo apt install docker-compose
    ```
    Note that we occassionally observed WSL manager installing outdated version of docker-compose on WSL. You can check it by running the following command:
    ```bash
    docker-compose --version 
    ```
    Any docker-compose version prior to 2.0 is not well supported within the MATRIX pipeline. Therefore if your version is older than v2.0 run the following:
    ```bash
    sudo apt update
    sudo apt upgrade docker-ce docker-ce-cli containerd.io

    # To re-check if your version is now updated
    docker-compose --version
    ```    
    If you stumble upon `socket permission denied` error, you can find a potential solution within the [common errors section](../../references/common_errors.md)
=== "Linux"

    ```bash
    # for ubuntu/Debian
    sudo apt install docker #installs docker desktop
    # for arch/manjaro. NOTE: need to explicitly install `docker-buildx`
    sudo pacman -Syu docker docker-compose docker-buildx
    ```


!!! Tip 
    The default settings of Docker have rather low resources configured, you might want to increase those in Docker desktop.
    

### Java

Our pipeline uses [Spark](https://spark.apache.org/) for distributed computations, which requires Java under the hood.

=== "MacOS"

    ```bash
    brew install openjdk@17
    brew link --overwrite openjdk@17 # makes the java version available in PATH
    ```

=== "Windows (WSL)"
    
    ```bash
    # install jdk
    sudo apt install openjdk-17-jdk
    ```

=== "Linux"

    ```bash
    # Java on Linux is complicated, check for your specific distro how to get JDK@17. 

    # On Arch/Manjaro
    pacman -S jdk17-openjdk
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

### Cloud-related tools
The following tools are related to technologies which link to cloud services (Google Cloud in our case). Although it is possible to run our pipeline without any dependencies on the cloud, we are utilizing Google Cloud Platform Storage and its resources heavily in our day-to-day work and therefore recommend installing them as well. 

Regular contributors are encouraged to also onboard to some of our GCP functionalities through service account - we get into more detail on this in the [deep dive section](../deep_dive/gcp_setup.md). Note that without service account, you will not be able to install the following tools successfully (which is totally fine for local development but blocking for cloud development).

!!! GCP onboarding
    Do you want to utilize our MATRIX system efficiently with real data & GCP? Please create [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) so that we can assist you in the best possible way.


#### gcloud SDK

We leverage Google (GCP) as our Cloud provider, the following cask installation provides CLI access to GCP, following this [Google tutorial](https://cloud.google.com/sdk/docs/install)

=== "MacOS"

    ```bash
    brew install --cask google-cloud-sdk
    ```
    
=== "Windows (WSL)"
    
    ```bash
    # check you have apt-transport-https and curl installed
    sudo apt-get install apt-transport-https ca-certificates gnupg curl
    # import Google Cloud public key
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    # add the gcloud CLI distribution URI as a package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    # update and install 
    sudo apt-get update && sudo apt-get install google-cloud-cli
    ```

After succesfully installation, authenticate the client:

```bash
gcloud auth login
gcloud auth application-default login
```

[Now, let's set up your environment :material-skip-next:](./local-setup.md){ .md-button .md-button--primary }
