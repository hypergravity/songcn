# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
# """
# Created on Mon Feb 13 12:35:08 2017
#
# @author: cham
# """
#
# #%%
# %pylab qt
# %matplotlib qt
# %load_ext autoreload
# %autoreload 2
#
#
# from song import master_
# from song import measure_xdrift
# from song import ccdproc_mod
# from song import utils
# import hrs
#
# from matplotlib import rcParams
# rcParams.update({'font.size':20})
# #%%
#
# reload(master_)
# reload(measure_xdrift)
#
# #%%
#
# # specify directory
# dp = '/hydrogen/song/star_spec/20170102/night/raw/'
#
# # scan fits files
# t = measure_xdrift.scan_files(dp)
#
# # measure cross-order drift
# t2, fig = measure_xdrift.check_xdrift(t)
# f90 = utils.scan_flux90(t, n_jobs=20, unit='adu')
# plot(f90[t['IMAGETYP']=='FLAT'])
# plot(f90[t['IMAGETYP']=='STAR'])
#
#
# """ BIAS """
# bias = master_.produce_master(t2, method='average', imagetp='BIAS', slc=slice(0, 10)).rot90(1)
#
# """ FLAT """
# flat = master_.produce_master(t2, method='average', imagetp='FLAT', slc=slice(0, 100)).rot90(1)
# flat_bias = ccdproc_mod.subtract_bias(flat, bias)
#
# """ THAR """
# thar = master_.produce_master(t2, method='median', imagetp='THAR', slc=slice(0, 1)).rot90(1)
# thar_bias = ccdproc_mod.subtract_bias(thar, bias)
#
#
#
# """ list all STAR images """
# master_.list_image(t2, imagetp='STAR', kwds=['OBJECT'])
#
#
#
# #%%
# """ VCUT """
# flat216 = ccdproc_mod.CCDData.read('/town/HRS/20161110/20161110006.fits', unit='adu')
# thar216 = ccdproc_mod.CCDData.read('/town/HRS/20161110/201611100039.fits', unit='adu')
# star216 = ccdproc_mod.CCDData.read('/town/HRS/20161110/201611100052.fits', unit='adu')
#
# fig = figure(figsize=(15, 10))
# ax = fig.add_subplot(211)
# ax.plot(flat_bias[:,1024])
# ax.plot(thar_bias[:,1024])
# ax.plot(star_bias[:,1024])
# ax.legend(['flat', 'thar', 'star'])
# ax.set_title('SONG')
#
# ax = fig.add_subplot(212)
# ax.plot(flat216[:,2048])
# ax.plot(thar216[:,2048])
# ax.plot(star216[:,2048])
# ax.legend(['flat', 'thar', 'star'])
# ax.set_title('2.16m HRS')
#
#
#
# #%%
#
# """ read noise """
# fps_bias_top10 = master_.list_image(t2, imagetp='BIAS', return_col='fps', kwds=['OBJECT']).data
#
# bias_top10 = []
# for fp in fps_bias_top10:
#     bias_top10.append(ccdproc_mod.CCDData.read(fp, unit='adu')).data
#
# bias_top10 = np.array(bias_top10)
# bias_top10_std = np.std(bias_top10, axis=0)
#
#
# imshow(bias_top10_std, interpolation='nearest')
#
#
#
# #%%
#
#
# # #################################### #
# #    find apertures
# # #################################### #
#
# #extract.list_image(t, 'FLAT')
#
# """ find apertures """
# find_aps_param_dict = dict(start_col=440, max_drift=8, max_apwidth=10,
#                            n_pix_goodap=100, n_adj=10, n_smooth=1, n_sep=3, c=5)
# ap_comb = hrs.combine_apertures([flat_bias], n_jobs=1, find_aps_param_dict=find_aps_param_dict)
# cheb_coefs, ap_uorder_interp = hrs.group_apertures(ap_comb, start_col=1024, order_dist=7)
#
# """ extract 1d FLAT & STAR & THAR """
# flat_bias_sl = hrs.substract_scattered_light(flat_bias, ap_uorder_interp, ap_width=10, shrink=.85)
# flat1d = hrs.extract_1dspec(flat_bias_sl, ap_uorder_interp, ap_width=8)[0]
#
# thar_bias_sl = hrs.substract_scattered_light(thar_bias, ap_uorder_interp, ap_width=10, shrink=.85)
# thar1d = hrs.extract_1dspec(thar_bias, ap_uorder_interp, ap_width=8)[0]
#
#
#
# # #################################### #
# #    determine wave_init
# # #################################### #
#
# # laod template thar
# thar_temp_path = '/home/cham/PycharmProjects/hrs/song/calibration/thar_template/thar_template.fits'
# wave_temp, thar_temp, order_temp = hrs.load_thar_temp(thar_temp_path)
#
# # fix thar
# thar1d_fixed = hrs.fix_thar_sat_neg(thar1d, arm=30, sat_count=60000*15)
# thar_temp_fixed = hrs.fix_thar_sat_neg(thar_temp, arm=30, sat_count=60000*15)
#
#
# """ initial estimation of wavelength """
# ytrim = (wave_temp.shape[0]*np.array([.2, .8])).astype(int)
# xtrim = (wave_temp.shape[1]*np.array([.2, .8])).astype(int)
# shift, corr2d = hrs.thar_corr2d(thar1d_fixed-np.percentile(thar1d_fixed, 50),
#                                 thar_temp-np.percentile(thar_temp, 50),
#                                 xtrim=xtrim, ytrim=ytrim,
#                                 y_shiftmax = 3, x_shiftmax=20)
#
# imshow(corr2d, interpolation='nearest')
# print ("@Cham: the shift is ", shift)
#
# wave_init = hrs.interpolate_wavelength_shift(wave_temp, shift, thar_temp, thar1d_fixed)
# order_init = hrs.interpolate_order_shift(order_temp, shift, thar1d_fixed)
#
#
# # #################################### #
# #    determine wave_init
# # #################################### #
#
# """ list all STAR images """
# master_.list_image(t2, imagetp='STAR', kwds=['OBJECT'])
#
#
# """ STAR """
# fig = figure()
#
# for i in range(63):
#     star = master_.produce_master(t2, method='median', imagetp='STAR', slc=slice(i, i + 1)).rot90(1)
#     star_bias = ccdproc_mod.subtract_bias(star, bias)
#
#     star_bias_sl = hrs.substract_scattered_light(star_bias, ap_uorder_interp, ap_width=10, shrink=.85)
#     star1d = hrs.extract_1dspec(star_bias_sl, ap_uorder_interp, ap_width=8)[0]
#
#     star1d_dblz = star1d/flat1d
#
#     #plot(wave_init.T, (flat1d/np.percentile(flat1d, 90, axis=1)[:, None]).T, 'b')
#     #plot(wave_init.T, (star1d/np.percentile(star1d, 90, axis=1)[:, None]).T, 'g')
#     plot(wave_init.T, (star1d_dblz/np.percentile(star1d_dblz, 90, axis=1)[:, None]).T*0.9+i, 'r')
#
#
#
# fig = figure()
# plot(wave_temp.T, thar_temp.T, 'r')
# plot(wave_init.T, thar1d.T, 'b')
#
#
# apflat = ccdproc_mod.CCDData.read('/home/cham/s2_2017-01-02T19-42-30_.fits', unit='adu')
# figure(), imshow(apflat, vmin=0.5, vmax=1.5)
#
#
# np.savetxt
#
#
#
#
#
#
# plot(wave_init.T, star1d_dblz.T, 'r')
#
# np.savetxt('/home/cham/Desktop/thar1d.dat', thar1d)
#
# thar_threshold = .90
#
# for i in range(thar1d.shape[0]):
#     plot(thar1d[i]/np.max(thar1d[i])+i)
#     ind_gt_thar_thre = np.where(thar1d[i]>np.percentile(thar1d[i], thar_threshold*100))[0]
#     plot(ind_gt_thar_thre, (thar1d[i]/np.max(thar1d[i]))[ind_gt_thar_thre]+i, 'r.')
#
#
# hist(thar1d[0], 1000)
# plot(thar1d[0])
#
# """  """
# thar_temp_path = '/home/cham/PycharmProjects/hrs/hrs/calibration/thar_template/thar_temp_w20160120022t.fits'
# wave_temp, thar_temp, order_temp = load_thar_temp(thar_temp_path)
#
#
#
#
# #%%
# for i in range(star1d_dblz.shape[0]):
#     plot(star1d_dblz[i, :]/np.median(star1d_dblz[i, :])+i*1)
#
#
# #%%
# imshow(log10(thar_bias), interpolation='nearest')
# plot(ap_comb.T, 'k-')
# plot(ap_uorder_interp.T, 'w-')
#
