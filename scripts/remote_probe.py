from __future__ import annotations

import argparse
from pathlib import Path

import paramiko


def connect(host: str, user: str) -> paramiko.SSHClient:
    key = paramiko.Ed25519Key.from_private_key_file(str(Path.home() / ".ssh" / "id_ed25519"))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, pkey=key, look_for_keys=False, allow_agent=False, timeout=10)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return code, out, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="REMOTE_GPU_HOST")
    parser.add_argument("--user", default="REMOTE_USER")
    args = parser.parse_args()
    client = connect(args.host, args.user)
    try:
        commands = {
            "identity": 'powershell -NoProfile -Command "$env:COMPUTERNAME; whoami"',
            "gpu": "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader",
            "torch": 'py -3.13 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\"NO_CUDA\\")"',
            "disk": 'powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used | Format-Table -AutoSize"',
        }
        for label, cmd in commands.items():
            code, out, err = run(client, cmd)
            print(f"--- {label} code={code} ---")
            print(out.strip())
            if err.strip():
                print("STDERR:", err.strip())
    finally:
        client.close()


if __name__ == "__main__":
    main()

