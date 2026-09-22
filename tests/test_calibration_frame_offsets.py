"""Frame offset files have one unambiguous camera/index convention."""

import json
from pathlib import Path
import tempfile
import unittest

from calibration_frame_offsets import load_frame_offsets, normalize_frame_offsets


class FrameOffsetTests(unittest.TestCase):
    def test_missing_cameras_default_to_zero_and_signed_offsets_are_preserved(self):
        self.assertEqual(normalize_frame_offsets({"cam2": -1, "cam4": 2}, 4),
                         {1: 0, 2: -1, 3: 0, 4: 2})
        self.assertEqual(normalize_frame_offsets({}, 2), {1: 0, 2: 0})
        self.assertEqual(load_frame_offsets(None, 2), {1: 0, 2: 0})

    def test_json_roundtrip_uses_integer_camera_ids_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.json"
            path.write_text(json.dumps({"cam1": 1, "cam3": -2}))
            self.assertEqual(load_frame_offsets(path, 3), {1: 1, 2: 0, 3: -2})

    def test_booleans_floats_fractions_strings_and_null_are_rejected(self):
        for value in (True, False, 1.0, 0.5, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    normalize_frame_offsets({"cam1": value}, 2)

    def test_unknown_or_noncanonical_camera_names_are_rejected(self):
        for key in (1, "1", "camera1", "cam0", "cam01", "cam-1", "cam1.0", "CAM1", "other"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "camera name"):
                    normalize_frame_offsets({key: 1}, 3)

    def test_out_of_range_camera_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside 1..3"):
            normalize_frame_offsets({"cam4": -1}, 3)

    def test_nonmapping_configuration_and_invalid_camera_counts_are_rejected(self):
        for mapping in (None, [], 1, "cam1"):
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    normalize_frame_offsets(mapping, 3)
        for count in (0, -1, True, 2.0):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    normalize_frame_offsets({}, count)


if __name__ == "__main__":
    unittest.main()
