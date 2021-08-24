import os


def main(source_dir, target_dir):
    """ Move all the files (recursively) from the source directory to a target directly.

    Args:
        source_dir:
        target_dir:

    Returns:

    """
    for root, dirs, files in os.walk(source_dir):

        if root == target_dir:
            # Avoid moving the same files twice
            # I.e. skip the target directory.
            continue

        for file in files:
            # TODO Move file to target dir

            filename = os.path.join(root, file)
            filename_target = os.path.join(target_dir, file)

            os.rename(filename, filename_target)

    return 1  # Succesfull


if __name__ == '__main__':
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    source = os.path.join(ROOT, 'data/raw/DH/newspapers/KB-JB837_LePeuple_sPDF-1886_UGent')
    target = source

    main(source, target)
