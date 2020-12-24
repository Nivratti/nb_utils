import os

def list_files(root_dir, mindepth = 1, maxdepth = float('inf'), filter_ext=[], return_relative_path=False):
    """
    Usage:

    d = get_all_files(rootdir, mindepth = 1, maxdepth = 2)

    This returns a list of all files of a directory, including all files in
    subdirectories. Full paths are returned.

    WARNING: this may create a very large list if many files exists in the 
    directory and subdirectories. Make sure you set the maxdepth appropriately.

    rootdir  = existing directory to start
    mindepth = int: the level to start, 1 is start at root dir, 2 is start 
               at the sub directories of the root dir, and-so-on-so-forth.
    maxdepth = int: the level which to report to. Example, if you only want 
               in the files of the sub directories of the root dir, 
               set mindepth = 2 and maxdepth = 2. If you only want the files
               of the root dir itself, set mindepth = 1 and maxdepth = 1
    
    filter_ext(list, optional) :  filter files ex. [.jpg, .png]
    return_relative_path(bool): Default false. If true return relative path else return absolute path
    """
    root_dir = os.path.normcase(root_dir)
    file_paths = []
    root_depth = root_dir.rstrip(os.path.sep).count(os.path.sep) - 1
    lowered_filter_ext = tuple([ext.lower() for ext in filter_ext])

    for abs_dir, dirs, files in sorted(os.walk(root_dir)):
        depth = abs_dir.count(os.path.sep) - root_depth
        if mindepth <= depth <= maxdepth:
            for filename in files:
                if filter_ext:
                    if not filename.lower().endswith(lowered_filter_ext):
                        continue

                if return_relative_path:
                    rel_dir = os.path.relpath(abs_dir, root_dir)
                    if rel_dir == ".":
                        file_paths.append(filename)
                    else:
                        file_paths.append(os.path.join(rel_dir, filename))
                else:
                    # append full absolute path
                    file_paths.append(os.path.join(abs_dir, filename))

        elif depth > maxdepth:
            # del dirs[:] 
            pass
    return file_paths