"""Native modelDNA byte source backed by the existing authenticated HubReader."""
import re
from threading import Lock
import time

from modeldna.io.source import ModelSource


class PinnedSource(ModelSource):
    is_remote = True

    def __init__(self, repo, revision, filenames, reader, pool, *, deadline, byte_cap):
        if not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError('A pinned revision is required')
        self.name, self.revision = repo, revision
        self.files = sorted(set(filenames))
        self.reader, self.pool = reader, pool
        self.deadline, self.byte_cap = deadline, byte_cap
        self.bytes_read = self.reserved_bytes = 0
        self.remotes, self.lock = {}, Lock()
        self.failed = False

    def list_files(self):
        return list(self.files)

    def _remote(self, filename):
        if filename not in self.files:
            raise KeyError('Unknown source file')
        with self.lock:
            self._admit()
            if filename not in self.remotes:
                self.remotes[filename] = self.reader.resolve(self.name, filename, self.revision)
            self._admit()
            return self.remotes[filename]

    def _admit(self):
        if self.failed:
            raise OSError('Source halted after a failed read')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Bounded run deadline')

    def size(self, filename):
        return self._remote(filename).size

    def read_range(self, filename, start, end):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Bounded run deadline')
        remote = self._remote(filename)
        if not 0 <= start <= end <= remote.size:
            raise ValueError('Range outside pinned file')
        with self.lock:
            self._admit()
            if self.reserved_bytes + end-start > self.byte_cap:
                raise ValueError('Native source byte cap exceeded')
            self.reserved_bytes += end-start
        try:
            self._admit()
            data = self.reader.read(remote, start, end)
        except Exception:
            with self.lock:
                self.failed = True
            raise
        with self.lock:
            self.bytes_read += len(data)
        return data

    def read_ranges(self, filename, ranges):
        return list(self.pool.map(lambda span: self.read_range(filename, *span), ranges))
