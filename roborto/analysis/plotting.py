"""Figure generation. Every report figure is produced here - never by hand."""

from __future__ import annotations

# Consistent look across all reports.
STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def apply_style() -> None:
    """Apply the shared matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


# TODO(claude-code): learning_curve(), performance_profile(), encoder_comparison_bar().
