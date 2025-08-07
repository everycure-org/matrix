import subprocess
import sys
from queue import Queue
from threading import Thread


def run_subprocess(
    cmd: str,
    check: bool = True,
    shell: bool = True,
    stream_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and handle errors.

    Args:
        cmd: Command string to execute
        check: If True, raise CalledProcessError on non-zero exit status
        shell: If True, execute the command through the shell
        stream_output: If True, capture and stream output to stdout/stderr.
                      If False, send output directly to system stdout/stderr.
    Returns:
        CompletedProcess instance with stdout/stderr (if stream_output=True)
    """
    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE if stream_output else None,
        stderr=subprocess.PIPE if stream_output else None,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    q = Queue()
    stdout, stderr = [], []

    if stream_output:
        tout = Thread(target=read_output, args=(process.stdout, [q.put, stdout.append]))
        terr = Thread(target=read_output, args=(process.stderr, [q.put, stderr.append]))
        twrite = Thread(target=write_output, args=(q.get,))
        for t in (tout, terr, twrite):
            t.daemon = True
            t.start()
        for t in (tout, terr):
            t.join()
        q.put(None)
        twrite.join()
    process.wait()
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, "".join(stdout) if stdout else None, "".join(stderr) if stderr else None
        )

    return subprocess.CompletedProcess(
        cmd, process.returncode, "".join(stdout) if stdout else None, "".join(stderr) if stderr else None
    )


def read_output(pipe, funcs):
    for line in iter(pipe.readline, ""):
        for func in funcs:
            func(line)
            # time.sleep(1)
    pipe.close()


def write_output(get):
    for line in iter(get, None):
        sys.stdout.write(line)


def command_exists(command: str) -> bool:
    """Check if a command exists in the system."""
    return run_subprocess(f"which {command}", check=False).returncode == 0
