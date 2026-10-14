"""Transport tests use invented opaque bytes, never real target events."""
import hashlib
import io
import json
from pathlib import Path
from queue import Queue
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

import newsforge
_staged = Path(__file__).resolve().parents[1] / "newsforge"
if str(_staged) not in newsforge.__path__:
    newsforge.__path__.insert(0, str(_staged))
from newsforge.external import acquire_parallel as module


class Response(io.BytesIO):
    def __init__(self, body, start=0, total=None, status=206, **changes):
        super().__init__(body)
        total = len(body) if total is None else total
        self.status = status
        self.headers = {"Content-Length": str(len(body)), "ETag": '"synthetic"',
            "Last-Modified": "synthetic", "Content-Range": f"bytes {start}-{start + len(body) - 1}/{total}"}
        self.headers.update(changes)
    def geturl(self):
        return module.DATA_URL


class ParallelTransportTests(unittest.TestCase):
    def test_ranges_cover_tail_exactly(self):
        self.assertEqual(module.byte_ranges(10, 4), [(0, 3), (4, 7), (8, 9)])

    def test_reject_status_offset_length_and_validator_changes(self):
        for changed in ({"status": 200}, {"Content-Range": "bytes 1-3/4"},
                        {"Content-Length": "2"}, {"ETag": '"new"'}, {"Last-Modified": "new"}):
            with self.subTest(changed=changed):
                response = Response(b"abcd", **changed)
                with self.assertRaises(ValueError):
                    module.validate_response(response, 0, 3, 4, '"synthetic"', "synthetic")

    def test_short_and_extra_bodies_fail_without_publishing_part(self):
        for body in (b"ab", b"abcde"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as folder:
                response = Response(body, **{"Content-Range": "bytes 0-3/4", "Content-Length": "4"})
                with patch.object(module.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises((IOError, ValueError)):
                        module.download_part(folder, 0, 3, 4, '"synthetic"', "synthetic", Queue(), threading.Event(), attempts=1)
                self.assertFalse(list(Path(folder).glob("*.bin")))
                self.assertTrue(list(Path(folder).glob("*.partial")))

    def test_assembly_rejects_gaps_overlaps_and_corruption_preserves_evidence(self):
        for start, digest in ((1, hashlib.sha256(b"ab").hexdigest()),
                              (0, hashlib.sha256(b"xx").hexdigest())):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "part").write_bytes(b"ab")
                record = {"start": start, "end_inclusive": start + 1, "bytes": 2,
                          "filename": "part", "sha256": digest}
                with self.assertRaises(ValueError):
                    module.assemble(root, [record], root / "archive", 2)
                self.assertFalse((root / "archive").exists())
                self.assertTrue((root / "archive.parallel.partial").exists())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "part").write_bytes(b"ab")
            record = {"start": 0, "end_inclusive": 1, "bytes": 2,
                      "filename": "part", "sha256": hashlib.sha256(b"ab").hexdigest()}
            with self.assertRaises(ValueError):
                module.assemble(root, [record, record], root / "archive", 4)

    def test_parallel_acquisition_opaque_bytes_and_double_hash_no_tar_read(self):
        payload = bytes(range(256)) * 600
        def request(req, timeout):
            if req.get_method() == "HEAD":
                return Response(b"", status=200, **{"Content-Length": str(len(payload))})
            value = req.get_header("Range").removeprefix("bytes=")
            start, end = map(int, value.split("-"))
            self.assertEqual(req.get_header("If-match"), '"synthetic"')
            return Response(payload[start:end + 1], start=start, total=len(payload))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(module, "ARCHIVE_BYTES", len(payload)), \
                 patch.object(module, "experiment_bindings", return_value={}), \
                 patch.object(module.urllib.request, "urlopen", side_effect=request), \
                 patch.object(tarfile, "open", side_effect=AssertionError("opaque acquisition must not parse")):
                result = module.acquire_parallel(root, root / "run", root / "archive", workers=2, chunk_bytes=65536)
            self.assertFalse(result["labels_inspected"])
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual((root / "archive").read_bytes(), payload)
            self.assertEqual(result["range_count"], 3)
            manifest = json.loads((root / "run/acquisition/ranges.json").read_text())
            self.assertEqual(sum(p["bytes"] for p in manifest["parts"]), len(payload))
            with self.assertRaises(FileExistsError):
                module.assemble(root / "run/acquisition/parts", manifest["parts"], root / "archive", len(payload))

    def test_malformed_range_writes_failure_and_never_publishes_archive(self):
        payload = b"x" * 65536
        def request(req, timeout):
            if req.get_method() == "HEAD":
                return Response(b"", status=200, **{"Content-Length": str(len(payload))})
            return Response(payload, status=200)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(module, "ARCHIVE_BYTES", len(payload)), \
                 patch.object(module, "experiment_bindings", return_value={}), \
                 patch.object(module.urllib.request, "urlopen", side_effect=request):
                with self.assertRaises(ValueError):
                    module.acquire_parallel(root, root / "run", root / "archive", workers=2, chunk_bytes=65536)
            self.assertFalse((root / "archive").exists())
            self.assertTrue((root / "run/acquisition/failure.json").exists())


if __name__ == "__main__":
    unittest.main()
