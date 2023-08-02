'''
Broad data-handling utilities, including get_files(f), which defines the folder naming structure.
'''

from pathlib import Path
from glob import glob

def get_files(f):
    '''
    Give a file path representing a single subject return all possible pipeline paths.

    Use this exclusively for calculating file paths as this enforces the naming convention.  Note, f can
    point to ANY data file in the data directory, or to a file that doesn't yet exist.  The dictionary
    it returns will contain all possible existing and future files that may be generated.  Essentially,
    this is the central repository for the file-mangling to make new outputs.

    Parameters
    ----------
    f : str
        Path to a file

    Returns
    -------
    dict : { str : str }
        { file_type : file_path}

    Example
    -------
    >>> files = get_files('../data/phantom_study-202201/raw/butterfly-iq/vertical-1.png')
    >>> print(files['annotated'])
    'C:\\src\\MTECTraumaticBrainInjury\\UltrasoundQuality\\python\\..\\data\\phantom_study-202201\\annotated\\butterfly-iq\\vertical-1-annotated.mha'
    '''
    # naming_map mostly defines the naming structure, though raw files may have an file extension and are a special-case
    naming_map = {
        'preprocessed' : ('preprocessed', '.mha'),
        'annotated_distance_map' : ('annotated', '-distance.mha'),
        'annotated_distance_map_points' : ('annotated', '-points.pickle'),
        'annotated_raw' : ('annotated_raw', '.seg.nrrd'),
        'annotated_snr' : ('annotated_raw', '-snr.json'),
        'annotated' : ('annotated', '-annotated.mha'),
        'annotated_component' : ('annotated', '-component.mha'),
        'registered' : ('registered', '-registered.mha'),
        'registered_transform' : ('registered', '-transform.h5'),
        'registered_metrics' : ('registered', '-metrics.pickle'),
        'registered_overlay' : ('registered', '-overlay.mha'),
        'registered_distancemap' : ('registered', '-distancemap.mha'),
        'registered_mask' : ('registered', '-mask.mha')
    }

    ans = {}
    subs = set([ x[0] for x in naming_map.values()]) 
    subs.add('raw') # all the subdirectories of the base data directory

    suffixes = [ x[1] for x in naming_map.values() ] # all the suffixes that can be added to a raw file's basename

    folder_type = None # what type of subfolder are we in
    p = Path(f).absolute()
    for x in p.parts[::-1]:
        if x in subs:
            folder_type = x
            break

    if folder_type is None:
        raise ValueError(f'Subtree of {f} must have a folder named one of {subs}')

    # since the raw file can have any file extension, we need to search for it 
    raw_search = _swap_and_append(f, folder_type, 'raw', '')
    raw_search = Path(remove_longest_suffix(str(raw_search), suffixes))
    tmp_glob = str(raw_search) + '*'
    tmp = glob(tmp_glob)
    if len(tmp) != 1:
        raise ValueError(f'Raw file must exist {tmp_glob}')
    raw = Path(tmp[0])
    base = remove_file_extension(str(raw))

    ans['raw'] = str(raw)
    for k, v in naming_map.items():
        ans[k] = str(_swap_and_append(base, 'raw', v[0], v[1]))

    return ans

def _swap_and_append(f, txt_match, txt_replace, suffix):
    '''
    Swaps the subfolder txt_match out of f with txt_replace, appends with suffix.

    Note, the file pointed to by f should not have any periods in its name.

    Parameters
    ----------
    f : str
        Path to file
    txt_match : str
        Name of subfolder to swap out
    txt_replace : str
        Name of subfolder to replace with
    suffix : str
        New suffix for file

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> f = './data/raw/foo/myfile.txt.zip'
    >>> swap_and_append(f, 'raw', 'preprocessed', '_pre.mha')
    WindowsPath('C:/resolved_path/data/preprocessed/foo/myfile_pre.mha')
    '''

    p = Path(f).absolute()
    idx = -(p.parts[::-1].index(txt_match)) - 1 # reverse the parts to find the last occurence of txt_match, returning negative index in original order
    return Path(str(Path(p.parts[0]).joinpath(*(p.parts[1:idx] + tuple([txt_replace]) + p.parts[idx+1:]))) + suffix)

def remove_file_extension(f):
    p = Path(f).absolute()
    base = p.stem.split('.')[0]
    return Path(p.parent).joinpath(base)

def remove_longest_suffix(txt, suffixes):
    '''
    Removes the longest match in suffixes from txt and returns the result.

    Parameters
    ----------
    txt : str
        txt to remove the longest suffix from
    suffixes : list of str
        list of suffixes, longest match will be removed from txt

    Returns
    -------
    str

    Examples
    --------
    >>> remove_longest_suffix('myfile-annotated.mha', ['.mha', '-annotated.mha', 'ted.mha'])
    'myfile'
    '''
    # take list, sort in descending order in length
    # i.e., if we have 'replace.mha' we want to match that
    # instead of '.mha'
    ss = sorted(suffixes, key=lambda x: len(x), reverse=True)

    n = len(txt)
    ans = txt
    for s in ss:
        r = txt.removesuffix(s)
        if len(r) != n:
            ans = r
            break

    return ans

