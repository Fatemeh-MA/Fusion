"""The one class map used everywhere in this repository.

Every .npy frame stores its label as an integer in column 3. If a source uses
a different order, remap it when converting (``tools/pcd_to_npy.py --label-map``)
so that the files on disk always follow this table.
"""

CLASS_NAMES = (
    "Robot",         # 0
    "Human",         # 1
    "Wall",          # 2
    "Floor",         # 3
    "Demonstrator",  # 4
    "Table",         # 5
    "AGV",           # 6
    "Unlabelled",    # 7
)
NUM_CLASSES = len(CLASS_NAMES)
UNLABELLED = 7

# RGB in [0, 1], same colours as the figures in the paper.
CLASS_COLORS = (
    (0.00, 0.00, 0.00),
    (169 / 255, 138 / 255, 14 / 255),
    (55 / 255, 58 / 255, 157 / 255),
    (224 / 255, 90 / 255, 46 / 255),
    (4 / 255, 72 / 255, 4 / 255),
    (150 / 255, 11 / 255, 150 / 255),
    (54 / 255, 4 / 255, 2 / 255),
    (1.00, 0.00, 0.00),
)
