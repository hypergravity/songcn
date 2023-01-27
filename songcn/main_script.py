from songcn import SongCalendar

sc = SongCalendar.from_dir(root="/Users/cham/projects/song/star_spec")
sc.pprint_all()

sn = sc.get_one_night("20191029")
print(f"SongNight contains {len(sc)} files")
sn.pprint_all()
print("Selected BIAS:", sn.select({"IMAGETYP": "BIAS"}, method="random"))
# sn.master_bias()
# sn.master_flat()


#%%

from song.song import Song
song = Song.init(
    rootdir="/Users/cham/projects/song/star_spec",
    date="20191030",
    jdnight=True,
    n_jobs=-1,
    verbose=True,
    subdir="night",
    node=2
)

