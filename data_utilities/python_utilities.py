"""Python utility functions."""

import difflib
import os
import re

try:
    from unidecode import unidecode
    HAS_UNIDECODE_MODULE = True
except ImportError:
    HAS_UNIDECODE_MODULE = False


def map_strings(set_keys,
                set_values,
                cutoff=0.8,
                ignore_no_matches=True):
    """Map a set of secondary strings to a set of primary strings."""
    N = 1
    CUTOFF = cutoff

    def get_matches(x):
        """Help to get matches."""
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


def process_string(x,
                   allow_symbols=False,  # underscore is not a symbol here.
                   allow_uppercase=False,
                   has_unidecode_module=HAS_UNIDECODE_MODULE):
    """Process string to lowercase and underscores only."""
    if has_unidecode_module:
        x = unidecode(x)
    if not allow_symbols:
        x = re.sub('\W', '_', x)
    if not allow_uppercase:
        x = x.lower()
    return x


def print_feature(feature, fill_length=79, fill_char='-'):
    """Print a feature in a nice and consistent way (aesthetics).

    Return None.

    """
    feature = feature + '   '
    print('\n',
          fill_length*'-',
          '\n{0:{1}<{2}s}\n'.format(feature, fill_char, fill_length),
          fill_length*'-',
          sep='')
    return None


if __name__ == '__main__':
    import doctest
    doctest.testmod()
