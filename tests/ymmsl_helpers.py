"""Helpers for loading ymmsl test configurations.

The in-process ``libmuscle.manager.manager.Manager`` requires a ymmsl **v0.2**
``Configuration`` (it calls ``config.root_model()``), but the test configs are
written in the v0.1 dialect. ``load_config`` merges the given v0.1
documents and converts the result to v0.2, mirroring what the
``muscle_manager`` CLI does in ``muscle3.muscle_manager.load_configuration``.
"""

import ymmsl
import ymmsl.v0_2 as v0_2


def load_config(*texts: str) -> v0_2.Configuration:
    """Load and merge ymmsl document(s) and return a v0.2 Configuration.

    Accepts v0.1 or v0.2 input; v0.1 documents are converted to v0.2.
    """
    config = ymmsl.load(texts[0])
    for text in texts[1:]:
        config.update(ymmsl.load(text))
    if isinstance(config, v0_2.Configuration):
        return config
    return ymmsl.convert_to(v0_2.Configuration, config)
