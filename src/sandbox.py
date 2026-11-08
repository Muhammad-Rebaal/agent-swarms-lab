"""In-memory sandboxed filesystem isolating agent I/O operations.

Uses pyfakefs to intercept filesystem operations entirely in volatile memory,
preventing agents from reading, writing, or discovering host operating system paths.
"""

import pyfakefs.fake_filesystem as fake_fs


class SandboxedFS:
    """Isolated in-memory filesystem container for multi-agent evaluation."""

    def __init__(self):
        """Initialize an empty in-memory filesystem with default experiment directories."""
        self.fs = fake_fs.FakeFilesystem()
        self.os_module = fake_fs.FakeOsModule(self.fs)
        self.open_func = fake_fs.FakeFileOpen(self.fs)

        self.fs.create_dir("/workspace")
        self.fs.create_dir("/shared_cache")

    def create_file(self, path: str, contents: str = "") -> None:
        """Create a new file in the sandboxed filesystem."""
        self.fs.create_file(path, contents=contents)

    def listdir(self, path: str) -> list[str]:
        """List entries within a directory in the sandbox."""
        return self.os_module.listdir(path)

    def read_file(self, path: str) -> str:
        """Read text contents from a sandboxed path."""
        with self.open_func(path, "r") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        """Write text content to a sandboxed path, creating parent directories as necessary."""
        parent = self.os_module.path.dirname(path)
        if parent and not self.fs.exists(parent):
            self.fs.create_dir(parent)
        if self.fs.exists(path):
            self.fs.remove_object(path)
        self.fs.create_file(path, contents=content)

    def exists(self, path: str) -> bool:
        """Check whether a path exists within the sandbox."""
        return self.fs.exists(path)
