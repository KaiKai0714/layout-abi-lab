"""Three-level residue evidence and evaluation of a conservative binary planner."""

from .evaluate import evaluate_index, render_markdown
from .features import DecisionFeatures, features_from_live, features_from_sizes
from .policies import LIVE_PLANNER_POLICIES, decide
