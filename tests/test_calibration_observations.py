"""Source-bound observation caches and globally consistent held-out frames."""

import os
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration_observations import (
    load_corner_cache,
    save_corner_cache,
    split_observations,
)


class CornerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.synced = self.base / "synced_raw"
        self.synced.mkdir()
        self.cache = self.base / "observations.npz"
        self.board_size = (3, 2)
        self.image_size = (160, 120)
        self.corners = np.array([(10. + x, 20. + y)
                                 for y in range(2) for x in range(3)], np.float32).reshape(6, 1, 2)
        self.observations = {
            cam: {f"frame_{frame:06d}.jpg": (self.corners + cam + frame, "temporary.jpg")
                  for frame in (0, 30, 60)}
            for cam in (1, 2)
        }
        self.scan_info = {cam: {"scanned": 3, "detected": 3, "every_n": 30}
                          for cam in (1, 2)}
        for cam in (1, 2):
            (self.synced / f"cam{cam}_synced.mp4").write_bytes(f"tiny-camera-{cam}".encode())
        self.save()

    def save(self):
        save_corner_cache(self.cache, self.observations, self.image_size,
                          self.board_size, self.scan_info, self.synced)

    def load(self, board_size=None, num_cams=2):
        return load_corner_cache(self.cache, self.synced,
                                 self.board_size if board_size is None else board_size,
                                 num_cams)

    def mutate_arrays(self, **replacements):
        with np.load(self.cache, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        arrays.update(replacements)
        with self.cache.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    def test_roundtrip_has_no_pickle_or_object_arrays(self):
        with np.load(self.cache, allow_pickle=False) as data:
            for name in data.files:
                self.assertFalse(data[name].dtype.hasobject, name)
        loaded, size, scan = self.load()
        self.assertEqual(size, self.image_size)
        self.assertEqual(scan, self.scan_info)
        self.assertEqual(loaded.keys(), self.observations.keys())
        for cam, rows in self.observations.items():
            self.assertEqual(loaded[cam].keys(), rows.keys())
            for name, (corners, _) in rows.items():
                restored, temporary_path = loaded[cam][name]
                np.testing.assert_array_equal(restored, corners)
                self.assertIsNone(temporary_path)
        # Loaded observations remain usable after the NPZ has been closed.
        loaded[1]["frame_000000.jpg"][0][0, 0, 0] = 999
        again, _, _ = self.load()
        self.assertNotEqual(again[1]["frame_000000.jpg"][0][0, 0, 0], 999)

    def test_changed_source_size_is_rejected(self):
        (self.synced / "cam1_synced.mp4").write_bytes(b"different-size-video")
        with self.assertRaisesRegex(ValueError, "source videos have changed"):
            self.load()

    def test_alignment_metadata_is_required_to_reuse_shifted_detections(self):
        # Logical frame 30 comes from physical source frame 29 in camera 2.
        rows = {cam: {name: value for name, value in frames.items()
                      if name != "frame_000000.jpg"}
                for cam, frames in self.observations.items()}
        save_corner_cache(self.cache, rows, self.image_size, self.board_size,
                          self.scan_info, self.synced, {1: 0, 2: -1})
        with self.assertRaisesRegex(ValueError, "frame offsets"):
            self.load()
        restored, _, _ = load_corner_cache(self.cache, self.synced, self.board_size,
                                           2, {1: 0, 2: -1})
        self.assertEqual(set(restored[2]), set(rows[2]))
        np.testing.assert_array_equal(restored[2]["frame_000030.jpg"][0],
                                      rows[2]["frame_000030.jpg"][0])

    def test_legacy_cache_implies_zero_offsets_and_cannot_silently_shift(self):
        with np.load(self.cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        metadata.pop("frame_offsets", None)
        self.mutate_arrays(metadata=np.array(json.dumps(metadata)))
        self.load()
        with self.assertRaisesRegex(ValueError, "frame offsets"):
            load_corner_cache(self.cache, self.synced, self.board_size, 2, {2: -1})

    def test_offset_cannot_produce_negative_physical_source_frame(self):
        save_corner_cache(self.cache, self.observations, self.image_size, self.board_size,
                          self.scan_info, self.synced, {2: -1})
        with self.assertRaisesRegex(ValueError, "Invalid cached observations"):
            load_corner_cache(self.cache, self.synced, self.board_size, 2, {2: -1})

    def test_changed_source_timestamp_is_rejected_even_when_size_is_same(self):
        path = self.synced / "cam1_synced.mp4"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(ValueError, "source videos have changed"):
            self.load()

    def test_board_orientation_and_camera_count_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "board/camera configuration"):
            self.load(board_size=(2, 3))
        for count in (1, 3):
            with self.subTest(camera_count=count):
                with self.assertRaisesRegex(ValueError, "board/camera configuration"):
                    self.load(num_cams=count)

    def test_duplicate_unsorted_negative_and_fractional_frames_are_rejected(self):
        for frames in ([0, 0, 60], [30, 0, 60], [-1, 30, 60], [0.1, 0.9, 60.1]):
            with self.subTest(frames=frames):
                self.save()
                self.mutate_arrays(cam1_frames=np.asarray(frames))
                with self.assertRaisesRegex(ValueError, "Invalid cached observations"):
                    self.load()

    def test_malformed_corner_shapes_and_nonfinite_values_are_rejected(self):
        proper = np.stack([self.corners] * 3)
        nan_values = proper.copy()
        nan_values[0, 0, 0, 0] = np.nan
        inf_values = proper.copy()
        inf_values[0, 0, 0, 1] = np.inf
        malformed = {
            "missing_axis": proper.reshape(3, 6, 2),
            "missing_corner": proper[:, :5],
            "extra_frame": np.stack([self.corners] * 4),
            "nan": nan_values,
            "infinity": inf_values,
        }
        for name, corners in malformed.items():
            with self.subTest(case=name):
                self.save()
                self.mutate_arrays(cam1_corners=corners)
                with self.assertRaisesRegex(ValueError, "Invalid cached observations"):
                    self.load()

    def test_object_arrays_cannot_trigger_pickle_loading(self):
        self.mutate_arrays(cam1_corners=np.array([{"not": "corners"}], dtype=object))
        with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
            self.load()

    def test_zero_frames_with_nonempty_corner_rows_are_rejected(self):
        self.mutate_arrays(cam1_frames=np.array([], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "Invalid cached observations"):
            self.load()

    def test_camera_without_detections_roundtrips(self):
        self.observations[2] = {}
        self.scan_info[2]["detected"] = 0
        self.save()
        loaded, size, scan = self.load()
        self.assertEqual(loaded[2], {})
        self.assertEqual(size, self.image_size)
        self.assertEqual(scan[2]["detected"], 0)


class ObservationSplitTests(unittest.TestCase):
    def test_every_fifth_union_key_is_held_out_consistently_across_cameras(self):
        names = [f"frame_{index * 30:06d}.jpg" for index in range(30)]
        rows = {
            1: {name: (np.array([index]), None) for index, name in enumerate(names)
                if index % 3 != 0},
            2: {name: (np.array([index]), None) for index, name in enumerate(names)
                if index % 4 != 0},
            3: {name: (np.array([index]), None) for index, name in enumerate(names)
                if index % 3 == 0},
            4: {},
        }
        train, validation, held_out = split_observations(rows)
        expected = names[4::5]
        self.assertEqual(held_out, expected)
        train_union = set().union(*(set(r) for r in train.values()))
        validation_union = set().union(*(set(r) for r in validation.values()))
        self.assertEqual(validation_union, set(expected))
        self.assertFalse(train_union & validation_union)
        self.assertEqual(train_union | validation_union, set(names))
        for camera, originals in rows.items():
            self.assertEqual(set(train[camera]) | set(validation[camera]), set(originals))
            self.assertFalse(set(train[camera]) & set(validation[camera]))
            self.assertEqual(set(validation[camera]), set(originals) & set(expected))
            self.assertEqual(set(train[camera]), set(originals) - set(expected))
            for name, value in originals.items():
                selected = validation[camera] if name in expected else train[camera]
                self.assertIs(selected[name], value)

    def test_short_or_empty_input_has_no_held_out_observations(self):
        for observations in ({}, {1: {}, 2: {}},
                             {1: {f"frame_{n:06d}.jpg": (n, None) for n in range(4)}, 2: {}}):
            with self.subTest(cameras=list(observations)):
                train, validation, held_out = split_observations(observations)
                self.assertEqual(train, observations)
                self.assertEqual(held_out, [])
                self.assertEqual(validation, {camera: {} for camera in observations})


if __name__ == "__main__":
    unittest.main()
