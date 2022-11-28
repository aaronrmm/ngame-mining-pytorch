import unittest

import pandas as pd

from ngame.ngame_dataset import generate_match_map


class MyTestCase(unittest.TestCase):
    def test_generate_match_map(self):
        test_df = pd.DataFrame()
        test_df["labels"] = ["zero", "one", "two", "three", "two", "four", "one"]
        match_map = generate_match_map(test_df, column_name_for_labels="labels")
        self.assertEqual([0, 1, 2, 3, 2, 4, 1], match_map)


if __name__ == "__main__":
    unittest.main()
