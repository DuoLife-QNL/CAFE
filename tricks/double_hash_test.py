import unittest

class DoubleHashKeyInfo:
    def __init__(self, hash_range):
        self.hash_range = hash_range

    def get_hash_keys(self, key):
        def hash_fn(x):
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
            x = (x ^ (x >> 27)) * 0x94d049bb133111eb
            x = x ^ (x >> 31)
            return x & 0xFFFFFFFFFFFFFFFF  # Ensure 64-bit unsigned integer

        key_1 = key % self.hash_range
        key_2 = hash_fn(key) % self.hash_range + self.hash_range
        
        return key_1, key_2

class TestDoubleHashKeyInfo(unittest.TestCase):
    def setUp(self):
        self.hash_range = 1000
        self.hasher = DoubleHashKeyInfo(self.hash_range)

    def test_get_hash_keys(self):
        test_keys = [1234552, 678190, 0, 999, 1000]
        for key in test_keys:
            key_1, key_2 = self.hasher.get_hash_keys(key)
            print(f"Key: {key}, Hash key 1: {key_1}, Hash key 2: {key_2}")
            
            # Test key_1
            self.assertGreaterEqual(key_1, 0)
            self.assertLess(key_1, self.hash_range)
            
            # Test key_2
            self.assertGreaterEqual(key_2, self.hash_range)
            self.assertLess(key_2, 2 * self.hash_range)
            
            # Test that key_1 and key_2 are different
            self.assertNotEqual(key_1, key_2)

    def test_consistency(self):
        key = 12345
        key_1_1, key_2_1 = self.hasher.get_hash_keys(key)
        key_1_2, key_2_2 = self.hasher.get_hash_keys(key)
        
        self.assertEqual(key_1_1, key_1_2)
        self.assertEqual(key_2_1, key_2_2)

if __name__ == '__main__':
    unittest.main()