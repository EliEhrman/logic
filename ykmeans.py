from __future__ import print_function
import numpy as np
import tensorflow as tf

import config

def make_db_cg(numrecs):
	v_db_norm = tf.Variable(tf.zeros([numrecs, config.c_key_dim], dtype=tf.float32), trainable=False)
	ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[numrecs, config.c_key_dim], name='ph_db_norm')
	op_db_norm_assign = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_assign')
	return v_db_norm, ph_db_norm, op_db_norm_assign

def make_per_batch_init_cg(numrecs, v_db_norm, num_centroids):
	# The goal is to cluster the convolution vectors so that we can perform dimension reduction
	# KMeans implementation
	# Intitialize the centroids indicies. Shape=[num_centroids]
	t_centroids_idxs_init = tf.random_uniform([num_centroids], 0, numrecs - 1, dtype=tf.int32,
											  name='t_centroids_idxs_init')
	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_centroids = tf.Variable(tf.zeros([num_centroids, config.c_key_dim], dtype=tf.float32), name='v_centroids')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))

	return v_centroids, op_centroids_init

def make_closest_idxs_cg(v_db_norm, v_all_centroids, num_db_seg_entries, num_tot_db_entries):
	ph_i_db_seg = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_db_seg')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	# op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
	# Do cosine distances for all centroids on all elements of the db. Shape [num_centroids, num_db_seg_entries]
	t_all_CDs = tf.matmul(v_all_centroids, v_db_norm[ph_i_db_seg*num_db_seg_entries:(ph_i_db_seg+1)*num_db_seg_entries, :], transpose_b=True, name='t_all_CDs')
	# For each entry in the chunk database, find the centroid that's closest.
	# Basically, we are finding which centroid had the highest cosine distance for each entry of the chunk db
	# This holds the index to the centroid which we can then use to create an average among the entries that voted for it
	# Shape=[num_db_seg_entries]
	t_closest_idxs_seg = tf.argmax(t_all_CDs, axis=0, name='t_closest_idxs_seg')
	# unconnected piece of cg building. Create a way of assigning np complete array back into the tensor Variable
	# code remains here because it would  be nice to replace with an in-graph assignment like TensorArray
	ph_closest_idxs = tf.placeholder(	shape=[num_tot_db_entries], dtype=tf.int32,
										name='ph_closest_idxs')
	v_closest_idxs = tf.Variable(tf.zeros(shape=[num_tot_db_entries], dtype=tf.int32),
								 name='v_closest_idxs')
	op_closest_idxs_set = tf.assign(v_closest_idxs, ph_closest_idxs, name='op_closest_idxs_set')

	return ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs

# Find the vote count and create an average for a single centroid
def vote_for_centroid_cg(v_db_norm, v_closest_idxs, num_tot_db_entries):
	# create placehoder to tell the call graph which iteration, i.e. which centroid we are working on
	ph_i_centroid = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_centroid')
	# Create an array of True if the closest index was this centroid
	# Shape=[num_centroids]
	t_vote_for_this = tf.equal(v_closest_idxs, ph_i_centroid, name='t_vote_for_this')
	# Count the number of trues in the vote_for_tis tensor
	# Shape=()
	t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
	# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
	# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_this_cluster = tf.where(t_vote_for_this, v_db_norm,
							  tf.zeros([num_tot_db_entries, config.c_key_dim]), name='t_this_cluster')
	# Sum the values for each property to get the aveage property
	# Shape=[c_kernel_size]
	t_cluster_sum = tf.reduce_sum(t_this_cluster, axis=0, name='t_cluster_sum')
	# Shape=[c_kernel_size]
	t_avg = tf.cond(t_vote_count > 0.0,
					lambda: tf.divide(t_cluster_sum, t_vote_count),
					lambda: tf.zeros([config.c_key_dim]),
					name='t_avg')

	return ph_i_centroid, t_avg, t_vote_count, t_vote_for_this

def update_centroids_cg(v_db_norm, v_all_centroids, v_closest_idxs, num_tot_db_entries, num_centroids):
	ph_new_centroids = tf.placeholder(dtype=tf.float32, shape=[num_centroids, config.c_key_dim], name='ph_new_centroids')
	ph_votes_count = tf.placeholder(dtype=tf.float32, shape=[num_centroids], name='ph_votes_count')
	# Do random centroids again. This time for filling in
	t_centroids_idxs = tf.random_uniform([num_centroids], 0, num_tot_db_entries - 1, dtype=tf.int32, name='t_centroids_idxs')
	# Shape = [num_centroids, c_kernel_size]
	# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
	t_new_centroids_norm = tf.nn.l2_normalize(ph_new_centroids, dim=1, name='t_new_centroids_norm')
	# Shape=[num_centroids]
	t_votes_count = ph_votes_count
	# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
	t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
	# Assign back to the original v_centroids so that we can go for another round
	op_centroids_update = tf.assign(v_all_centroids, tf.where(tf.greater(ph_votes_count, 0.0), t_new_centroids_norm,
														  t_centroids_from_idxs, name='centroids_where'),
									name='op_centroids_update')

	# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
	# its centroid.
	# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
	# replicate the centroid of that cluster to the locations alligned with each member of the database
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_centroid_broadcast = tf.gather(v_all_centroids, v_closest_idxs, name='t_centroid_broadcast')
	# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
	# Shape=[num_tot_db_entries]
	t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
	# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers. Shape=()
	t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')
	return t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count





def cluster_db(sess, numrecs_uneven, t_y_db_uneven, num_centroids):
	uneven_remainder = numrecs_uneven % config.c_kmeans_num_db_segs
	numrecs = numrecs_uneven - uneven_remainder
	t_y_db = t_y_db_uneven[:numrecs]
	v_db_norm, ph_db_norm, op_db_norm_assign = make_db_cg(numrecs)

	v_centroids, op_centroids_init = make_per_batch_init_cg(numrecs, t_y_db, num_centroids)


	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_all_centroids = tf.Variable(tf.zeros([config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim],
										   dtype=tf.float32), name='v_centroids')
	ph_all_centroids = tf.placeholder(dtype=tf.float32,
									  shape=[config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim],
									  name='ph_all_centroids')
	op_all_centroids_set = tf.assign(v_all_centroids, ph_all_centroids, name='op_all_centroids_set')

	ph_random_centroids = tf.placeholder(dtype=tf.float32,
										 shape=[config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim],
										 name='ph_random_centroids')
	ph_centroid_sums  = tf.placeholder(dtype=tf.float32,
									   shape=[config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim],
									   name='ph_centroid_sums')
	ph_count_sums  = tf.placeholder(dtype=tf.float32,
									shape=[config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim],
									name='ph_count_sums')
	t_new_centroids = tf.where(ph_count_sums > 0.0, ph_centroid_sums/ph_count_sums, ph_random_centroids)
	op_all_centroids_norm_set = tf.assign(v_all_centroids, tf.nn.l2_normalize(t_new_centroids, dim=1),
										  name='op_all_centroids_norm_set')

	# cg to create the closest_idxs for one segment of one batch of the v_db
	ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs \
		= make_closest_idxs_cg(	v_db_norm, v_all_centroids,
								num_db_seg_entries = numrecs / config.c_kmeans_num_db_segs,
								num_tot_db_entries = numrecs)
	# Create cg that calculates the votes for just one centroid, must be fed the index of the centroid to calculate for
	ph_i_centroid, t_avg, t_vote_count, t_vote_for_this \
		= vote_for_centroid_cg(	v_db_norm, v_closest_idxs,
								num_tot_db_entries = numrecs)

	t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count \
		= update_centroids_cg(	v_db_norm, v_all_centroids, v_closest_idxs,
								num_tot_db_entries = numrecs * config.c_kmeans_num_batches,
								num_centroids = config.c_num_clusters * config.c_kmeans_num_batches)


	nd_all_controids = np.zeros([config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim], dtype=np.float32)
	# We're pretending we are creating the y_db in batches.Right now there is only one batch
	# and the	y_db is passed in	from the outside.
	for ibatch in range(config.c_kmeans_num_batches):
		nd_y_db = sess.run(t_y_db)
		sess.run(op_db_norm_assign, feed_dict={ph_db_norm:nd_y_db })
		nd_all_controids[ibatch*config.c_num_clusters:(ibatch+1)*config.c_num_clusters] = sess.run(op_centroids_init)
		print('building initial db. ibatch=', ibatch)
	sess.run(op_all_centroids_set, feed_dict={ph_all_centroids:nd_all_controids})

	for iter_kmeans in range(config.c_kmeans_iters):
		l_centroid_avgs = []
		l_centroid_counts = []
		l_kmeans_err = []
		for ibatch in range(config.c_kmeans_num_batches):
			nd_y_db = sess.run(t_y_db)
			sess.run(op_db_norm_assign, feed_dict={ph_db_norm: nd_y_db})
			for iseg in range(config.c_kmeans_num_db_segs):
				n1 = sess.run(t_closest_idxs_seg, feed_dict={ph_i_db_seg:iseg})
				if iseg == 0:
					nd_closest_idxs = n1
				else:
					nd_closest_idxs = np.concatenate([nd_closest_idxs, n1], axis=0)
			sess.run(op_closest_idxs_set, feed_dict={ph_closest_idxs:nd_closest_idxs})
			nd_new_centroids = np.ndarray(dtype = np.float32, shape = [config.c_num_clusters * config.c_kmeans_num_batches, config.c_key_dim])
			nd_votes_count = np.ndarray(dtype = np.float32, shape = [config.c_num_clusters * config.c_kmeans_num_batches])
			for icent in range(config.c_num_clusters * config.c_kmeans_num_batches):
				r_cent_avg, r_cent_vote_count = sess.run([t_avg, t_vote_count], feed_dict={ph_i_centroid:icent})
				nd_new_centroids[icent, : ]  = r_cent_avg
				nd_votes_count[icent] = r_cent_vote_count
			r_votes_count, r_centroids, r_kmeans_err \
				= sess.run(	[t_votes_count, op_centroids_update, t_kmeans_err],
							feed_dict={ph_new_centroids:nd_new_centroids, ph_votes_count:nd_votes_count})
			l_centroid_avgs.append(r_centroids)
			l_centroid_counts.append(r_votes_count)
			l_kmeans_err.append(r_kmeans_err)
			print('building kmeans db. ibatch=', ibatch)
		np_centroid_avgs = np.stack(l_centroid_avgs)
		np_centroid_counts = np.stack(l_centroid_counts)
		np_count_sums = np.tile(np.expand_dims(np.sum(np_centroid_counts, axis=0), axis=-1), reps=[1, config.c_key_dim])
		np_br_centroid_counts = np.tile(np.expand_dims(np_centroid_counts, axis=-1), reps=[1, config.c_key_dim])
		np_centroid_facs = np.multiply(np_centroid_avgs, np_br_centroid_counts)
		np_centroid_sums = np.sum(np_centroid_facs, axis=0)
		np.random.shuffle(nd_all_controids)
		r_centroids = sess.run(	op_all_centroids_norm_set,
								feed_dict={	ph_random_centroids: nd_all_controids,
											ph_centroid_sums: np_centroid_sums,
											ph_count_sums: np_count_sums})
		print('kmeans iter:', iter_kmeans, 'kmeans err:', np.mean(np.stack(l_kmeans_err)))


	for iseg in range(config.c_kmeans_num_db_segs):
		n1 = sess.run(t_closest_idxs_seg, feed_dict={ph_i_db_seg: iseg})
		if iseg == 0:
			nd_closest_idxs = n1
		else:
			nd_closest_idxs = np.concatenate([nd_closest_idxs, n1], axis=0)

	return nd_closest_idxs