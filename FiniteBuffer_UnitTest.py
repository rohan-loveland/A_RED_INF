import unittest
from FiniteBuffer import FiniteBuffer

class TestInit(unittest.TestCase):
    def setUp(self):
        self.FiniteBuffer = FiniteBuffer(10)

    def test_btree_ratio_set_up(self):
        self.assertEqual(self.FiniteBuffer.ball_tree_interval, 10 * .8)

    def test_num_ball_trees_set_up(self):
        self.assertEqual(self.FiniteBuffer.num_ball_trees, 2)

    def test_ball_tree_list_set_up(self):
        self.assertEqual(len(self.FiniteBuffer.ball_trees) == 0 )

    def test_buffer_size_set_up(self):
        self.assertEqual(self.FiniteBuffer.data_circular_buffer.size, 10)
        self.assertEqual(self.FiniteBuffer.label_circular_buffer.size, 10)
        #self.assertEqual(self.FiniteBuffer.cluster_id_circular_buffer.size, 10) # removed feature
        self.assertEqual(self.FiniteBuffer.relevance_circular_buffer.size, 10)
        self.assertEqual(self.FiniteBuffer.buffer_size, 10)

if __name__ == '__main__':
    unittest.main()
