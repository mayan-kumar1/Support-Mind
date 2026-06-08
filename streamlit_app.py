import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Just re-export the UI app
from ui.app import *
