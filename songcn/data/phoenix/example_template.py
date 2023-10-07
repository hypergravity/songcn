import joblib
from astropy.io import fits

wave_path = "songcn/data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
flux_path = (
    "songcn/data/phoenix/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
)

wave_cut_path = "songcn/data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011-cut.joblib"
flux_cut_path = (
    "songcn/data/phoenix/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes-cut.joblib"
)

wave = fits.getdata(wave_path)
flux = fits.getdata(flux_path)

ind_cut = (wave >= 5000) & (wave <= 8000)
wave_cut = wave[ind_cut]
flux_cut = flux[ind_cut]

# plt.plot(wave_cut, flux_cut)

joblib.dump(
    wave_cut,
    wave_cut_path,
)
joblib.dump(
    flux_cut,
    flux_cut_path,
)
