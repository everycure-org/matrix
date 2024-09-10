---
title: Installation
---

# Installation

This page assumes basic knowledge of the following technologies. We will provide instructions how to install them but as a quick reference, below are the necessary tools:

!!! info "Support on Windows, MacOS and Linux"
    We are mostly using MacOS but try our best to provide an onboarding for all
    platforms. This guide assumes Our guide assumes usage of [homebrew](https://brew.sh/)
    to manage packages on MacOS, [Windows
    WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) usage on Windows
    and some system proficiency for Linux users. If you find your platform could be
    better supported, do [send a
    PR](https://github.com/everycure-org/matrix/edit/main/src/docs/src/onboarding/index.md)!

### Windows Subsystem for Linux (WSL)

If you are running on Windows, you need to install Windows Subsystem for Linux as the following steps require a UNIX OS. You can follow this [tutorial from Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install). 

!!! Tip 
    If using WSL, you need to ensure the MATRIX Github repo is cloned within WSL.

=== Windows (Powershell)
   
    ```bash
    wsl --install
    ```

After installing WSL, you need to restart your computer.

### Cloning Github repos in WSL

WSL does not allow cloning repos by HTTPS so cloning requires using SSH, following the Github tutorials on [generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux) and [adding a new SSH key to your account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

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

### Python (Mac)

We advise managing your Python installation using [`pyenv`](https://github.com/pyenv/pyenv).

=== "MacOS"

    ```bash
    brew install pyenv
    ```
    
### Python (WSL)

Steps for installing pyenv in WSL following the [tutorial on Github](https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv)

1. Install dependencies (if not already installed):

   ```bash
   sudo apt-get update; sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
   tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
   ```
   
2. Clone the pyenv repository:

   ```bash
   git clone https://github.com/pyenv/pyenv.git ~/.pyenv
   ```
   
3. Define the PYENV_ROOT environment variable:

   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   ```
   
4. Enable pyenv init:

   ```bash
   echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
   ```
   
5. Restart your shell so the changes take effect:

   ```bash
   exec "$SHELL"
   ```

6. *Check the pyenv version*:

   ```bash
   pyenv --version
   ```
   
   This should print the version of pyenv that you have installed, for example: ```bash pyenv 2.3.6.```
   
After following these steps, you should have pyenv installed and ready to use on your WSL environment.

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
    # install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # create virtual environment
    uv venv
    # activate virtual environment
    source .venv/bin/activate
    
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

=== "WSL"
    
    ```bash
    # install docker
    sudo apt install docker
    # install docker-compose
    sudo apt install docker-compose
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

=== "WSL"
    
    ```bash
    # install jdk
    sudo apt install openjdk-11-jdk
    ```

=== "Linux"

    ```bash
    # Java on Linux is complicated, check for your specific distro how to get JDK@11. 

    # On Arch/Manjaro
    pacman -S jdk11-openjdk
    ```

### gcloud SDK

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
