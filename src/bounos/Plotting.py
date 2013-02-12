import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
matplotlib.use("WXAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button

from __init__ import Metric

import numpy as np


def interactive_plot(data):
	"""
	Generate the MPL data browser for the flock data
	"""
	# Generate Arrangements for viewport + accessory views
	plt.close('all')
	fig = plt.figure()
	gs = GridSpec(9, 16)
	ax = plt.subplot(gs[:-1, 1:], projection = '3d')
	axH = plt.subplot(gs[:, 0])
	#    axB = plt.subplot(gs[-1,1:])

	# Find initial display state for viewport
	lines = [ax.plot(xs, ys, zs)[0] for xs, ys, zs in data.p]
	for n, line in enumerate(lines):
		line.set_label(data.names[n])

	#Configure the Time Slider
	timeax = plt.axes([0.2, 0.0, 0.65, 0.03])
	timeslider = Slider(timeax, 'Time', 0, data.tmax, valinit = 0)

	#Configure the buttons
	playax = plt.axes([0.8, 0.025, 0.1, 0.04])
	play = Button(playax, 'Play', hovercolor = '0.975')

	# Set initial Vector Display
	vectors = data.heading_slice(0)
	count = len(vectors)
	ind = np.arange(count)
	width = 0.11

	global rectX, rectY, rectZ
	rectX = axH.barh(ind, tuple([vec[0] for vec in vectors]), width, color = 'r')
	rectY = axH.barh(ind + width, tuple([vec[1] for vec in vectors]), width, color = 'g')
	rectZ = axH.barh(ind + 2 * width, tuple([vec[2] for vec in vectors]), width, color = 'b')


	def press(event):
		if event.key is 'left':
			timeslider.set_val(timeslider.val - 1)
		elif event.key is 'right':
			timeslider.set_val(timeslider.val + 1)
		else:
			print('press', event.key)


	fig.canvas.mpl_connect('key_press_event', press)

	def update_viewport(val):
		"""
		Update Line display across time
		"""
		for n, line in enumerate(lines):
			(xs, ys, zs) = data.trail_of(n, timeslider.val)
			line.set_data(xs, ys)
			line.set_3d_properties(zs)
			line.set_label(data.names[n])

	def update_headings(val):
		"""
		Update Vector Heading display across time
		"""
		vectors = data.heading_slice(timeslider.val)
		axH.cla()
		rectX = axH.barh(ind, tuple([vec[0] for vec in vectors]), width, color = 'r')
		rectY = axH.barh(ind + width, tuple([vec[1] for vec in vectors]), width, color = 'g')
		rectZ = axH.barh(ind + 2 * width, tuple([vec[2] for vec in vectors]), width, color = 'b')

		axH.set_ylabel("Vector")
		axH.set_yticks(ind + width)
		axH.set_yticklabels(data.names)

	def update(val):
		if timeslider.val > data.tmax:
			timeslider.set_val(data.tmax)
			return
		if timeslider.val < 0:
			timeslider.set_val(0)
			return

		update_viewport(val)
		update_headings(val)
		plt.draw()

	timeslider.on_changed(update)

	shape = data.environment
	ax.legend()
	ax.set_title("Tracking overview of %s" % data.title)
	ax.set_xlim3d((0, shape[0]))
	ax.set_ylim3d((0, shape[1]))
	ax.set_zlim3d((0, shape[2]))
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()


def KF_metric_plot(metric):
	assert isinstance(metric, Metric), "Not a metric: %s" % metric
	assert hasattr(metric, 'data'), "No Data"

	import numpy as np
	from bounos import DataPackage

	from pykalman import KalmanFilter

	data = DataPackage("/dev/shm/dat-2013-02-01-13-58-48.aietes.npz")

	rnd = np.random.RandomState(0)

	# generate a noisy sine wave to act as our fake observations
	n_timesteps = data.tmax
	x = range(0, n_timesteps)
	records = metric.data

	try:
		obs_dim = len(records[0])
	except TypeError as e:
		obs_dim = 1

	observations = np.ma.array(records) # to put it as(tmax,3)
	masked = 0
	for i in x:
		try:
			if rnd.normal(2, 2) >= 0:
				observations[i] = np.ma.masked
				masked += 1
		except BaseException as e:
			print(i)
			raise e

	print("%f%% Masked" % ((masked * 100.0) / data.tmax))

	print("Records: Shape: %s, ndim: %s, type: %s" % (records.shape, records.ndim, type(records)))

	# create a Kalman Filter by hinting at the size of the state and observation
	# space.  If you already have good guesses for the initial parameters, put them
	# in here.  The Kalman Filter will try to learn the values of all variables.
	kf = KalmanFilter(n_dim_obs = obs_dim, n_dim_state = obs_dim)

	# You can use the Kalman Filter immediately without fitting, but its estimates
	# may not be as good as if you fit first.

	#states_pred = kf.em(observations, n_iter=data.tmax).smooth(observations)
	#print 'fitted model: %s' % (kf,)

	# Plot lines for the observations without noise, the estimated position of the
	# target before fitting, and the estimated position after fitting.
	fig = plt.figure(figsize = (16, 6))
	ax1 = fig.add_subplot(111)
	filtered_state_means = np.zeros((n_timesteps, kf.n_dim_state))
	filtered_state_covariances = np.zeros((n_timesteps, kf.n_dim_state, kf.n_dim_state))

	for t in x:
		if t == 0:
			tmp = np.zeros(kf.n_dim_state)
			tmp.fill(500)
			filtered_state_means[t] = tmp
			print(filtered_state_means[t])
			continue

		if masked and not observations.mask[t].any():
			ax1.axvline(x = t, linestyle = '-', color = 'r', alpha = 0.1)
		try:
			filtered_state_means[t], filtered_state_covariances[t] = \
				kf.filter_update(filtered_state_means[t - 1], filtered_state_covariances[t - 1], observations[t])
		except IndexError as e:
			print(t)
			raise e
	(p, v) = (filtered_state_means, filtered_state_covariances)
	errors = map(np.linalg.norm, p[:] - records[:])

	pred_err = ax1.plot(x, p[:], marker = ' ', color = 'b',
	                    label = 'predictions-x')
	obs_scatter = ax1.plot(x, records, linestyle = '-', color = 'r',
	                       label = 'observations-x', alpha = 0.8)
	ax1.set_ylabel(metric.label)
	ax2 = ax1.twinx()
	error_plt = ax2.plot(x, errors, linestyle = ':', color = 'g', label = "Error Distance")
	ax2.set_yscale('log')
	ax2.set_ylabel("Error")
	lns = pred_err + obs_scatter + error_plt
	labs = [l.get_label() for l in lns]
	plt.legend(lns, labs, loc = 'upper right')
	plt.xlim(0, 500)
	ax1.set_xlabel('time')

	fig.suptitle(data.title)

	print("Predicted ideal %s: %s" % (metric.label, str(p[-1])))

	plt.show()
