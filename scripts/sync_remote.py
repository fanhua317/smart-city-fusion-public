from __future__ import annotations

import argparse
import os
from pathlib import Path

import paramiko


def connect(host: str, user: str) -> paramiko.SSHClient:
    key = paramiko.Ed25519Key.from_private_key_file(str(Path.home() / ".ssh" / "id_ed25519"))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, pkey=key, look_for_keys=False, allow_agent=False, timeout=10)
    return client


def win_path(path: str) -> str:
    return path.replace("/", "\\")


def ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = win_path(path).split("\\")
    cur = parts[0]
    for part in parts[1:]:
        if not part:
            continue
        cur += "\\" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def put_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    count = 0
    ensure_remote_dir(sftp, remote)
    for root, dirs, files in os.walk(local):
        root_path = Path(root)
        rel = root_path.relative_to(local)
        rdir = win_path(remote if str(rel) == "." else remote + "\\" + str(rel))
        ensure_remote_dir(sftp, rdir)
        for file in files:
            lp = root_path / file
            rp = rdir + "\\" + file
            sftp.put(str(lp), rp)
            count += 1
    return count


def get_tree(sftp: paramiko.SFTPClient, remote: str, local: Path) -> int:
    local.mkdir(parents=True, exist_ok=True)
    count = 0

    def walk(rdir: str, ldir: Path) -> None:
        nonlocal count
        ldir.mkdir(parents=True, exist_ok=True)
        for item in sftp.listdir_attr(rdir):
            rp = rdir + "\\" + item.filename
            lp = ldir / item.filename
            if item.st_mode & 0o040000:
                walk(rp, lp)
            else:
                sftp.get(rp, str(lp))
                count += 1

    walk(remote, local)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="REMOTE_GPU_HOST")
    parser.add_argument("--user", default="REMOTE_USER")
    parser.add_argument("--remote-root", default=r"D:\fusion_rank_work\smart_city_fusion")
    parser.add_argument("--direction", choices=["up", "down"], default="up")
    args = parser.parse_args()
    client = connect(args.host, args.user)
    sftp = client.open_sftp()
    try:
        if args.direction == "up":
            ensure_remote_dir(sftp, args.remote_root)
            n1 = put_tree(sftp, Path("target"), args.remote_root + r"\target")
            n2 = put_tree(sftp, Path("scripts"), args.remote_root + r"\scripts")
            print(f"uploaded target={n1} scripts={n2} remote={args.remote_root}")
        else:
            n = get_tree(sftp, args.remote_root + r"\results", Path("remote_sync/results"))
            print(f"downloaded results={n}")
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()

