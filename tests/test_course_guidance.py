"""The beginner route must cover the curriculum without prerequisite cycles."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from course_guidance import MAP, BEGINNER_STAGES


class CourseGuidance(unittest.TestCase):
    def test_complete_topological_beginner_route(self):
        route = [day for _, days in BEGINNER_STAGES for day in days]
        self.assertEqual(sorted(route), list(range(1, 55)))
        self.assertEqual(sorted(row["day"] for row in MAP), list(range(1, 55)))
        position = {day: i for i, day in enumerate(route)}
        for row in MAP:
            for prerequisite in row["prerequisites"]:
                self.assertLess(position[prerequisite], position[row["day"]], row)
            answer = ROOT / f"docs/solutions/day-{row['day']:02d}.md"
            text = answer.read_text(encoding="utf-8")
            self.assertEqual(text.count("<details>"), 3)
            self.assertEqual(text.count("</details>"), 3)
            self.assertTrue((ROOT / row["path"]).is_file())
