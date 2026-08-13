"""
Makes relgt and relgnn modules properly accessible.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
for module in ["relgt", "relgnn"]:
    sys.path.append(os.path.join(current_dir, "external", module))
