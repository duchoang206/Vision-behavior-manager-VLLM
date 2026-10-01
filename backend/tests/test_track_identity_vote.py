import os
import tempfile
import unittest

import numpy as np

from core.track_identity_service import TrackIdentityService
from core.track_identity_vote import TrackIdentityVoter

NOISE = np.random.RandomState(0)


def appearance(seed, noise=.05):
    base = np.random.RandomState(seed).randn(512)
    return base / np.linalg.norm(base) + noise * NOISE.randn(512) / np.sqrt(512)


def voter_with_two_robots():
    voter = TrackIdentityVoter(store_path=None)
    for _ in range(6):
        voter.add_prototype("Robot_2001", appearance(1, .3))
        voter.add_prototype("Robot_6868", appearance(2, .3))
    return voter


class TrackIdentityVoteTests(unittest.TestCase):
    def test_wrong_detector_label_is_corrected_only_after_majority(self):
        voter = voter_with_two_robots()
        key = ("cam", 7)
        decisions = [voter.observe(key, "Robot_6868", appearance(1), now=index) for index in range(7)]
        self.assertTrue(all(decision is None for decision in decisions))  # < min_votes: keep the model label
        self.assertEqual("Robot_2001", voter.observe(key, "Robot_6868", appearance(1), now=8))

    def test_agreeing_label_is_never_overridden(self):
        voter = voter_with_two_robots()
        for index in range(15):
            self.assertIsNone(voter.observe(("cam", 1), "Robot_2001", appearance(1), now=index))

    def test_one_odd_frame_does_not_flip_the_label(self):
        voter = voter_with_two_robots()
        key = ("cam", 3)
        for index in range(12):
            voter.observe(key, "Robot_6868", appearance(1), now=index)
        self.assertEqual("Robot_2001", voter.decision(key))
        voter.observe(key, "Robot_6868", appearance(2), now=13)
        self.assertEqual("Robot_2001", voter.decision(key))

    def test_unknown_appearance_does_not_correct(self):
        voter = voter_with_two_robots()
        for index in range(15):
            voter.observe(("cam", 4), "Robot_6868", appearance(99, 0), now=index)
        self.assertIsNone(voter.decision(("cam", 4)))

    def test_label_outside_the_gallery_is_never_relabeled(self):
        voter = voter_with_two_robots()
        for index in range(15):
            voter.observe(("cam", 8), "Robot_1", appearance(1), now=index)
        self.assertIsNone(voter.decision(("cam", 8)))

    def test_needs_two_registered_robots(self):
        voter = TrackIdentityVoter(store_path=None)
        voter.add_prototype("Robot_2001", appearance(1))
        self.assertIsNone(voter.classify(appearance(1)))

    def test_prototypes_round_trip_and_skip_duplicates(self):
        path = os.path.join(tempfile.mkdtemp(), "prototypes.json")
        voter = TrackIdentityVoter(store_path=path)
        vector = appearance(1, 0)
        self.assertTrue(voter.add_prototype("Robot_2001", vector))
        self.assertFalse(voter.add_prototype("Robot_2001", vector))
        voter.save()
        self.assertEqual(["Robot_2001"], TrackIdentityVoter(store_path=path).labels())

    def test_service_relabels_robots_but_not_other_classes(self):
        voter = voter_with_two_robots()
        for index in range(10):
            voter.observe(("cam", 5), "Robot_6868", appearance(1), now=index)
        service = TrackIdentityService(voter=voter)
        objects = [dict(id=5, category="robot", label="Robot_6868"), dict(id=9, category="rack", label="Rack")]
        service.apply("cam", objects)
        self.assertEqual("Robot_2001", objects[0]["label"])
        self.assertEqual("Robot_6868", objects[0]["model_label"])
        self.assertTrue(objects[0]["reid_corrected"])
        self.assertEqual("Rack", objects[1]["label"])


if __name__ == "__main__":
    unittest.main()
