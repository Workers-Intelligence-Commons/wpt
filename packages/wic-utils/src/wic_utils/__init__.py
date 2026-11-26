"""Small helpers every part shares (the depot, the freight, the dock and
WPT), one submodule each, none of them tied to one part:

    wic_utils.programs   the outside programs a part runs, with a clear error
                         when one is missing, and the repository commit
    wic_utils.files      the sha256 of a file, the one way a part hashes one,
                         and a lock the operating system holds on a file
    wic_utils.parallel   how much parallel work this machine takes, measured
                         again before each new item, and items run that way
                         with their results in the items' order
    wic_utils.lines      the form of every printed line: its time, its level,
                         its text
    wic_utils.web        every web request, under the program name wpt
    wic_utils.env        the .env file a part reads its settings from beside
                         the environment

Import the submodule you need: this package itself imports nothing.
"""
