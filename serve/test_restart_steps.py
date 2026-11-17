"""QuantLab F (11/17): StrataEngine.restart() loosens the start step by step after an engine died mid-conversation
(RESTART_STEPS_DEFAULT): waits for the VRAM, then a larger VRAM reserve, then a smaller RAM budget - never a shorter
context or another K/V format; a start with a nearly empty expert cache on a card of 6 GiB or more counts as failed
except on the last step.  A fake engine: no process, no GPU."""
import unittest
from unittest import mock

import serve.server as S


class _Proc:
    def wait(self, timeout=None):
        return 1


_SEAT_LOG = None


def _seat_log():
    """A log whose start says what the engine says when its expert cache gets no VRAM."""
    global _SEAT_LOG
    if _SEAT_LOG is None:
        import tempfile
        fd, _SEAT_LOG = tempfile.mkstemp(suffix=".log")
        with open(fd, "w", encoding="utf-8") as f:
            f.write("strata generate: no VRAM is left for the expert cache: it needs at least 461 slots (706 MiB)\n"
                    "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache)\n")
    return _SEAT_LOG


class _Fake(S.StrataEngine):
    plan: list = []
    calls: list = []

    def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False, cpus=None, _attempt=0):
        self.spawn = (exe, list(args), cwd, log, env, False, cpus)
        self.info = {}
        if lazy:
            return
        _Fake.calls.append((list(args), dict(env or {})))
        mode = _Fake.plan.pop(0) if _Fake.plan else "ok"
        self.proc = _Proc()
        self.log_path, self.log_start = None, 0
        if mode == "seat":                       # the start fails for want of expert-cache slots (the engine's words)
            self.log_path = _seat_log()
            raise S.EngineDied("fake: did not start (no VRAM for the expert cache)")
        if mode == "fail":
            raise S.EngineDied("fake: did not start")
        self.info["expert_slots"] = 56 if mode == "slow" else 1121

    def close(self):
        pass


ARGS = ["--pack", "p", "--max-context", "65536", "--kv", "k8v4", "--vram-reserve-mib", "1024",
        "--resident-budget-gib", "16"]
ENV = {"STRATA_PARTIAL_PIN_GIB": "17", "X": "1"}


def _arg(args, k):
    return args[args.index(k) + 1] if k in args else None


@mock.patch.object(S, "RESTART_VRAM_WAIT_S", 0.3)
@mock.patch.object(S.StrataEngine, "RESTART_RETRY_S", 0.01)
@mock.patch("builtins.print")
class RestartSteps(unittest.TestCase):
    def engine(self, steps=None, card=8188):
        e = _Fake("strata.exe", ARGS, None, None, ENV, lazy=True)
        e.restart_steps = steps
        e.vram_now = (lambda: (1500, card)) if card else None
        _Fake.calls.clear()
        return e

    def test_second_try_same_settings(self, _p):
        e = self.engine()
        _Fake.plan = ["fail", "ok"]
        e.restart()
        self.assertEqual([a for a, _ in _Fake.calls], [ARGS, ARGS])

    def test_reserve_then_budget_never_context(self, _p):
        e = self.engine()
        _Fake.plan = ["fail"] * 4
        with self.assertRaises(S.EngineDied):
            e.restart()
        self.assertEqual([_arg(a, "--vram-reserve-mib") for a, _ in _Fake.calls], ["1024", "1024", "1536", "2048"])
        self.assertEqual(_arg(_Fake.calls[3][0], "--resident-budget-gib"), "12.75")
        self.assertEqual(_Fake.calls[3][1]["STRATA_PARTIAL_PIN_GIB"], "13.5")
        self.assertEqual(_Fake.calls[3][1]["X"], "1")
        for a, _ in _Fake.calls:
            self.assertEqual(_arg(a, "--max-context"), "65536")
            self.assertEqual(_arg(a, "--kv"), "k8v4")

    def test_later_restart_begins_at_the_step_that_started(self, _p):
        e = self.engine()
        _Fake.plan = ["fail", "fail", "ok"]
        e.restart()
        _Fake.plan = ["ok"]
        e.restart()
        self.assertEqual(len(_Fake.calls), 4)
        self.assertEqual(_arg(_Fake.calls[3][0], "--vram-reserve-mib"), "1536")

    def test_slow_start_tries_the_next_step(self, _p):
        e = self.engine()
        _Fake.plan = ["slow", "ok"]
        e.restart()
        self.assertEqual(len(_Fake.calls), 2)
        self.assertEqual(e.info["expert_slots"], 1121)

    def test_last_slow_start_is_kept(self, _p):
        e = self.engine()
        _Fake.plan = ["slow"] * 4
        e.restart()
        self.assertEqual(len(_Fake.calls), 4)
        self.assertEqual(e.info["expert_slots"], 56)

    def test_small_card_and_no_reader_do_not_judge_slots(self, _p):
        for card in (4096, None):
            e = self.engine(card=card)
            _Fake.plan = ["slow"]
            e.restart()
            self.assertEqual(len(_Fake.calls), 1)

    def test_no_steps_is_the_old_same_command(self, _p):
        e = self.engine(steps=[])
        _Fake.plan = ["fail", "fail", "ok"]
        e.restart()
        self.assertEqual([a for a, _ in _Fake.calls], [ARGS, ARGS, ARGS])

    # 11/17 case A: a start short of expert-cache slots lowers the reserve instead (1024 -> 512 -> 256), the RAM budget
    # stays; after the last try the reason is seat_short (the chat screen can suggest a lighter model)
    def test_seat_short_lowers_the_reserve(self, _p):
        e = self.engine()
        _Fake.plan = ["seat", "seat", "ok"]
        e.restart()
        self.assertEqual([_arg(a, "--vram-reserve-mib") for a, _ in _Fake.calls], ["1024", "1024", "512"])
        self.assertTrue(all(_arg(a, "--resident-budget-gib") == "16" for a, _ in _Fake.calls))
        self.assertIsNone(e.restart_reason)
        _Fake.plan = ["ok"]                      # a later restart goes on in the same direction
        e.restart()
        self.assertEqual(_arg(_Fake.calls[-1][0], "--vram-reserve-mib"), "512")

    def test_seat_short_every_try_says_why(self, _p):
        e = self.engine()
        _Fake.plan = ["seat"] * 4
        with self.assertRaises(S.EngineDied) as cm:
            e.restart()
        self.assertIn("seat_short", str(cm.exception))
        self.assertEqual(e.restart_reason, "seat_short")
        self.assertEqual([_arg(a, "--vram-reserve-mib") for a, _ in _Fake.calls], ["1024", "1024", "512", "256"])
        for a, _ in _Fake.calls:
            self.assertEqual(_arg(a, "--max-context"), "65536")
            self.assertEqual(_arg(a, "--kv"), "k8v4")

    def test_slow_starts_lower_the_reserve_and_the_last_is_noted(self, _p):
        e = self.engine()
        _Fake.plan = ["slow"] * 4
        e.restart()
        self.assertEqual([_arg(a, "--vram-reserve-mib") for a, _ in _Fake.calls], ["1024", "1024", "512", "256"])
        self.assertEqual(e.restart_reason, "seat_short")
        self.assertEqual(e.info.get("restart_note"), "seat_short")

    def test_other_failure_then_seat_short(self, _p):
        e = self.engine()
        _Fake.plan = ["fail", "seat", "ok"]
        e.restart()
        self.assertEqual([_arg(a, "--vram-reserve-mib") for a, _ in _Fake.calls], ["1024", "1024", "512"])

    def test_lowering_never_raises_a_small_reserve(self, _p):
        args, _, said = S.restart_loosened(["--vram-reserve-mib", "200"], None, {"vram_reserve_mib": "-512"})
        self.assertEqual(args, ["--vram-reserve-mib", "200"])
        self.assertEqual(said, [])


class RestartStepsConfig(unittest.TestCase):
    def test_reader(self):
        self.assertEqual(S.restart_steps_of({}), S.RESTART_STEPS_DEFAULT)
        self.assertEqual(S.restart_steps_of({"restart_steps": []}), [])
        self.assertEqual(S.restart_steps_of({"restart_steps": False}), [])
        with self.assertRaises(ValueError):
            S.restart_steps_of({"restart_steps": [{"max_context": "32768"}]})


if __name__ == "__main__":
    unittest.main()
