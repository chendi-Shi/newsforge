"""Portable test fixtures that inherit the workspace's Windows ACLs."""

from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import uuid


@contextmanager
def workspace_temporary_directory(prefix="newsforge-test-"):
    """Create an isolated direct child of tests; remove only that exact path.

    A normal mkdir inherits workspace permissions. Python's TemporaryDirectory
    requests mode 0700, which can create an inaccessible owner-only Windows ACL
    when the test runner uses a restricted token.
    """
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("temporary-directory prefix must be a simple filename prefix")
    tests_directory = Path(__file__).resolve().parent
    directory = tests_directory / f"{prefix}{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        resolved = directory.resolve()
        if (not resolved.is_absolute() or resolved.parent != tests_directory
                or resolved != directory):
            raise RuntimeError("refusing to remove a temporary path outside its original tests directory")
        shutil.rmtree(resolved)
