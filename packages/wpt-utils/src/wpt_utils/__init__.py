"""Small helpers only WPT uses, one submodule each:

    wpt_utils.device     the most capable PyTorch device on this machine,
                         and the processor's name (imports torch)
    wpt_utils.settings   a plug-in's settings, typed from the defaults its file declares

The helpers every part shares are wic_utils's. Import the submodule you
need: this package itself imports nothing, so wpt_utils.settings works where
PyTorch is not installed.
"""
