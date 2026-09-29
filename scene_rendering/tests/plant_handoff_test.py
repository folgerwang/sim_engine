"""Frame protocol of the plant cluster hand-off (drawable path <-> cluster path).

Run with python -m unittest discover -s <this directory> -p '*_test.py'.

The expand draws, in frame N, the jobs the G-buffer compaction appended in
frame N-1.  The drawable passes must therefore skip exactly THAT set.  The
old rule skipped the set handed off in frame N itself; over budget the
winners are atomic-order random, so the two sets differed every frame and
trees vanished / doubled (the "corrupt vertex buffer" flicker).  This models
both rules with the same stamp encoding nt_instance_compact.comp uses.
"""
import random
import unittest


def next_frame(f):
    return (f % 65534) + 1          # ClusterRenderer::advancePlantHandoffFrame


class Stamps:
    """handoff_stamp[]: two words per instance, word = frame << 16 | bit."""

    def __init__(self, n):
        self.w = [0] * (2 * n)

    def owned(self, inst, tpl, drawn):
        if drawn == 0:
            return False
        v = self.w[inst * 2 + (drawn & 1)]
        return (v >> 16) == drawn and (v & (1 << (tpl & 15))) != 0

    def stamp(self, inst, tpl, frame):
        i = inst * 2 + (frame & 1)
        v = self.w[i]
        bit = 1 << (tpl & 15)
        self.w[i] = (v | bit) if (v >> 16) == frame else ((frame << 16) | bit)


def run(new_rule, frames=400, seed=7):
    rng = random.Random(seed)
    n_inst = 300
    # (instance, template) records; some instances carry two records
    # (e.g. trunk + canopy meshes sharing one instance list).
    recs = [(i, i % 12) for i in range(n_inst)] + \
           [(i, 12 + i % 3) for i in range(0, n_inst, 7)]
    stamps = Stamps(n_inst)
    queued = []                      # jobs appended by the last compaction
    primed = False
    handoff_frame = drawn_frame = drew = 0
    errors = 0
    for n in range(frames):
        if rng.random() < 0.03:      # placed sync re-finalized the templates
            primed = False
        # recordPlantExpand
        drew = handoff_frame if primed else 0
        cluster = set(queued) if primed else set()
        primed = True
        # advancePlantHandoffFrame + setPlantHandoff
        drawn_frame, handoff_frame = drew, next_frame(handoff_frame)
        # G-buffer compaction: visible records, budget winners random
        visible = [r for r in recs if rng.random() < 0.8]
        order = visible[:]
        rng.shuffle(order)
        budget = 40
        queued = []
        drawable = set()
        for r in order:
            inst, tpl = r
            owned = stamps.owned(inst, tpl, drawn_frame)
            handed = budget > 0
            if handed:
                budget -= 1
                queued.append(r)
                stamps.stamp(inst, tpl, handoff_frame)
            skip = owned if new_rule else handed
            if not skip:
                drawable.add(r)
        for r in visible:
            count = (r in drawable) + (r in cluster)
            if count != 1:
                errors += 1
    return errors


class PlantHandoffProtocol(unittest.TestCase):
    def test_old_rule_flickers_over_budget(self):
        self.assertGreater(run(new_rule=False), 1000)

    def test_every_visible_record_drawn_exactly_once(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                self.assertEqual(run(new_rule=True, seed=seed), 0)

    def test_frame_ids_alternate_parity_across_wrap(self):
        f = 65533
        for _ in range(4):
            g = next_frame(f)
            self.assertNotEqual(f & 1, g & 1)
            self.assertTrue(1 <= g <= 65534)
            f = g


if __name__ == "__main__":
    unittest.main()
