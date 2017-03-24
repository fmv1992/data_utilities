"""Python helper functions."""

import difflib
import os
import re


def map_strings(set_keys,
                set_values,
                cutoff=0.8,
                ignore_no_matches=True):
    u"""Map a set of secondary strings to a set of primary strings."""
    N = 1
    CUTOFF = cutoff

    def get_matches(x):
        u"""Helper to get matches."""
        result_list = difflib.get_close_matches(
            x, set_values, n=N, cutoff=CUTOFF)
        if ignore_no_matches:
            if result_list:
                return result_list[0]
            else:
                return ''
        else:
            return result_list[0]
    mapper = map(lambda x: (x, get_matches(x)),
                 set_keys)
    return dict(mapper)


def list_matching_files_in_path(regex_pattern, path):
    """Produce an interable of files matching 'regex_pattern' in path.

    Returns:
        generator: a generator of files that match 'regex_pattern'.

    """
    path = os.path.abspath(path)
    walk = os.walk(path)
    matching_files = map(lambda x: (x[0], x[1],
                                    tuple(y for y in x[2] if
                                          re.search(regex_pattern, y) is not
                                          None)),
                         walk)
    for root, subdir, files in matching_files:
        if files:
            for f in files:
                yield os.path.abspath(os.path.join(root, f))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
