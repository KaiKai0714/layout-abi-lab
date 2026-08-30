"""Prepare writable cache paths in minimal or UID-mapped containers."""

from __future__ import annotations

import getpass
import os
import pwd
from pathlib import Path
from typing import Any


def prepare_runtime(prefix: str = "layoutabi") -> dict[str, Any]:
    """Create writable framework caches and tolerate a UID missing from passwd.

    Some cluster containers map the caller's numeric UID without adding a passwd
    entry. PyTorch, Triton, and profiler code may call ``pwd.getpwuid`` while creating
    caches. This process-local compatibility shim supplies only the current identity.
    """

    uid = os.getuid()
    root = Path(os.environ.get("LAYOUTABI_RUNTIME_HOME", f"/tmp/{prefix}_{uid}"))
    defaults = {
        "HOME": root / "home",
        "XDG_CACHE_HOME": root / "xdg",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "TRITON_CACHE_DIR": root / "triton",
        "TMPDIR": root / "tmp",
    }
    for key, default in defaults.items():
        current = Path(os.environ.get(key, str(default)))
        if key == "HOME" and (current == Path("/") or not os.access(current, os.W_OK)):
            current = default
        os.environ[key] = str(current)
        current.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("USER", os.environ.get("LAYOUTABI_RUNTIME_USER", prefix))
    os.environ.setdefault("LOGNAME", os.environ["USER"])

    result: dict[str, Any] = {
        "uid": uid,
        "home": os.environ["HOME"],
        "user": os.environ["USER"],
        "passwd_missing": False,
        "patched_getpwuid": False,
    }
    try:
        pwd.getpwuid(uid)
        return result
    except KeyError:
        result["passwd_missing"] = True

    fake = pwd.struct_passwd(
        (os.environ["USER"], "x", uid, uid, "", os.environ["HOME"], "/bin/sh")
    )
    real_getpwuid = pwd.getpwuid
    real_getpwnam = pwd.getpwnam

    def getpwuid(lookup_uid: int) -> pwd.struct_passwd:
        try:
            return real_getpwuid(lookup_uid)
        except KeyError:
            if lookup_uid == uid:
                return fake
            raise

    def getpwnam(name: str) -> pwd.struct_passwd:
        try:
            return real_getpwnam(name)
        except KeyError:
            if name in {os.environ["USER"], os.environ.get("LOGNAME", "")}:
                return fake
            raise

    pwd.getpwuid = getpwuid  # type: ignore[method-assign]
    pwd.getpwnam = getpwnam  # type: ignore[method-assign]
    getpass.getuser = lambda: os.environ["USER"]  # type: ignore[method-assign]
    result["patched_getpwuid"] = True
    return result

