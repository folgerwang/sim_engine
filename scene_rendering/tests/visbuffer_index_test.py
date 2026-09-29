"""Check material triangle addressing against indexed-draw semantics.

Run with python -m unittest discover -s <this directory> -p '*_test.py'.
The expression comes from the production shader; fixtures include a dynamic
plant relocation where interpreting firstIndex as bytes escapes its vertices.
"""
from pathlib import Path
import re
import unittest

SHADER = Path(__file__).resolve().parents[2] / "shaders/visbuffer_material.comp"


def shader_base(first_index, primitive):
    source = SHADER.read_text(encoding="utf-8")
    expression = re.search(r"uint base = ([^;]+);", source).group(1)
    expression = expression.replace("di.index_offset", str(first_index))
    expression = expression.replace("prim_id", str(primitive))
    expression = re.sub(r"(\d+)u\b", r"\1", expression)
    if not re.fullmatch(r"[\d\s()+*<>/-]+", expression):
        raise AssertionError("Review changed shader addressing expression")
    return eval(expression, {"__builtins__": {}}, {})


class VisibilityTriangleAddressing(unittest.TestCase):
    def test_nonzero_cluster_offsets_match_raster_triangles(self):
        indices = list(range(90))
        for first_index in (3, 9, 12, 33):
            for primitive in range(4):
                with self.subTest(first_index=first_index, primitive=primitive):
                    base = shader_base(first_index, primitive)
                    expected = indices[first_index:][primitive * 3:primitive * 3 + 3]
                    self.assertEqual(indices[base:base + 3], expected)

    def test_relocated_plant_stays_in_its_vertex_range(self):
        # Other meshes precede a two-triangle template at firstIndex 9.
        indices = [2, 4, 6, 20, 22, 24, 40, 42, 44, 100, 101, 102, 102, 103, 100]
        template_first, dynamic_first = 100, 1000
        for primitive, expected in enumerate(((1000, 1001, 1002), (1002, 1003, 1000))):
            base = shader_base(9, primitive)
            vertices = tuple(i + dynamic_first - template_first for i in indices[base:base + 3])
            self.assertEqual(vertices, expected)
            self.assertTrue(all(dynamic_first <= i < dynamic_first + 4 for i in vertices))


if __name__ == "__main__":
    unittest.main()
