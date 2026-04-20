"""
Custom fsspec LocalFileSystem that places temporary files in the target
directory instead of the system /tmp.  This avoids failures when /tmp is
on a small partition while the actual checkpoint directory has plenty of
space.

Usage – call ``register()`` once at program start (e.g. in your training
script) to replace the default ``file://`` / ``local://`` fsspec handlers:

    from ptp.atomic_fs import register
    register()

After that, PyTorch Lightning's fsspec-backed atomic saves will write
temporary files alongside the final destination rather than under /tmp.
"""

import os
import tempfile

from fsspec.implementations.local import LocalFileOpener, LocalFileSystem
from fsspec.registry import register_implementation


class SameDirFileOpener(LocalFileOpener):
    """Like LocalFileOpener but writes the staging temp file next to the
    final destination instead of in the system temp directory."""

    def _open(self):
        if self.f is None or self.f.closed:
            if self.autocommit or "w" not in self.mode:
                # Read-mode or auto-commit: delegate to parent implementation.
                super()._open()
                return

            # Non-autocommit write: create temp file in the target directory
            # so we stay on the same filesystem/partition.
            target_dir = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(target_dir, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=target_dir)
            os.close(fd)  # we want a normal buffered open
            self.temp = name
            self.f = open(name, mode=self.mode)

            if "w" not in self.mode:
                self.size = self.f.seek(0, 2)
                self.f.seek(0)
                self.f.size = self.size


class SameDirLocalFileSystem(LocalFileSystem):
    """LocalFileSystem that stages atomic writes in the target directory."""

    protocol = ("file", "local")

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir and "w" in mode:
            self.makedirs(self._parent(path), exist_ok=True)
        return SameDirFileOpener(path, mode, fs=self, **kwargs)


def register(clobber: bool = True) -> None:
    """Register SameDirLocalFileSystem as the handler for file:// and local://.

    Parameters
    ----------
    clobber:
        Whether to overwrite any existing registration (default True).
    """
    for protocol in ("file", "local"):
        register_implementation(protocol, SameDirLocalFileSystem, clobber=clobber)
