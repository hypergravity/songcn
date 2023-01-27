import glob


def glob_fits(root="/Users/cham/projects/song/star_spec"):
    """ Glob .fits files in a root directory and sort. """
    fps = glob.glob(root + "/**/*.fits", recursive=True)
    fps.sort()
    return fps


def test():
    fps = glob_fits()
    print(f"{len(fps)} files found!")
    print(fps)
