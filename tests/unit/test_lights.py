import threading
import time
import unittest

from lights import LightController


class FakeParam:
    def __init__(self):
        self.calls = []

    def set_value(self, key, value):
        self.calls.append((key, value))


class FakeCf:
    def __init__(self, uri):
        self.link_uri = uri
        self.param = FakeParam()


class FakeScf:
    def __init__(self, uri):
        self.cf = FakeCf(uri)


class LightControllerThreadingTests(unittest.TestCase):
    def test_run_emotion_sync_waits_for_sequence_start_and_then_updates_lights(self):
        uri = "radio://test"
        controller = LightController()
        controller.enabled[uri] = True
        controller.segments = [
            {"start": 0.0, "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        ]
        scf = FakeScf(uri)

        worker = threading.Thread(target=controller.run_emotion_sync, args=(scf,))
        worker.start()

        time.sleep(0.05)
        self.assertEqual(scf.cf.param.calls, [])

        controller.set_sequence_start(time.perf_counter())
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            scf.cf.param.calls,
            [("ring.effect", "1"), ("ring.effect", "0")],
        )

    def test_stop_releases_waiting_light_thread_without_setting_effect(self):
        uri = "radio://test"
        controller = LightController()
        controller.enabled[uri] = True
        controller.segments = [
            {"start": 0.0, "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        ]
        scf = FakeScf(uri)

        worker = threading.Thread(target=controller.run_emotion_sync, args=(scf,))
        worker.start()

        time.sleep(0.05)
        controller.stop()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(scf.cf.param.calls, [])


if __name__ == "__main__":
    unittest.main()
