# SSHing into a Vertex AI workbench: a guide for lost data scientists

This is a guide for how to connect to a Vertex AI workbench from your VSCode or Cursor code editor.

1. Install the Remote - SSH extension. 
2. Open your SSH config file. An easy way to do this is:
    1. Press Cmd-Shift-p (on Mac) to open the command palette  
    2. Select “SSH-Remote: Open SSH Configuration File” command
    3. Select your SSH file, which should be listed. For instance, the directory of mine is “Users.alexei/.ssh/config"
3. Modify the config file as follows, making the appropriate changes to the config file:
    
    ```coffeescript
    Host alexei-dev-workbench
        # Paste Instance ID from VM instance details here
        HostName compute.XXXXXXXXXXXXX
        # paste username
        # Same username as if you connected via gcloud command directly, e.g.
        # $> gcloud compute ssh --zone "us-central1-a" "alexei-dev-workbench" --tunnel-through-iap --project "<project_id>"
        User alexei_everycure_org
        IdentityFile /Users/alexei/.ssh/google_compute_engine
        # change machine name, project ID & maybe zone
        ProxyCommand /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/Caskroom/google-cloud-sdk/520.0.0/google-cloud-sdk/lib/gcloud.py compute start-iap-tunnel 'alexei-dev-workbench' %p --listen-on-stdin --project <project_id> --zone us-central1-a
        CheckHostIP no
        HashKnownHosts no
        # same as above for Hostname
        HostKeyAlias compute.XXXXXXXXXXX
        IdentitiesOnly yes
        StrictHostKeyChecking no
    ```
    
    Note that this is specific to Mac, hence the homebrew directories for the packages for the ProxyCommand. For this to work, you need to check that the Python and google-cloud-sdk version you have installed matches up to those listed in the config. If you find that there is no directory `google-cloud-sdk` in `/opt/homebrew/Caskroom,` run the command `brew install google-cloud-sdk` . 
    
4. Run the command “Remote-SSH: connect to a Host” in the command palette. You should see the host specified in the config file in my case alexei-dev-workbench. Select it. 
5. Congratulations, you should be connected to your machine!
Couple notes:
    1. The `home`  directory contains two user directories `jupyter` and `<your User name>`. You will not have rights make modifications in the `jupyter`  directory so work in the other one. 
    2. To work with Jupyter notebooks, you have to find the Jupyter extension in the Marketplace and “Install in SSH:  <your host name>”.