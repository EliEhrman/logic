from __future__ import print_function
import sys
import learn
import els

def learn_orders_success(init_pl, status_pl, orders_pl, results_pl, all_the_dicts, db_len_grps, el_set_arr, sess, learn_vars):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	init_db , status_db, orders_db, results_db = \
		els.make_rec_list(init_pl), els.make_rec_list(status_pl), els.make_rec_list(orders_pl), els.make_rec_list(results_pl),
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	full_db = init_db + status_db
	num_orders = len(orders_db)
	event_step_id = learn_vars[0]
	for iorder in range(num_orders):
		event_step_id += 1
		order = orders_db.pop(0)
		out_str = ''
		out_str = els.print_phrase(order.phrase(), order.phrase(), out_str, def_article_dict)
		print('New order: ', out_str)
		out_str = ''
		out_str = els.print_phrase(results_pl[iorder], results_pl[iorder], out_str, def_article_dict)
		print('Result of order: ', out_str)
		learn.learn_one_story_step(full_db+orders_db, order.phrase(), cascade_els, [results_pl[iorder]],
								   def_article_dict, db_len_grps, el_set_arr, glv_dict, sess, event_step_id)
		orders_db.append(order)

	learn_vars[0] = event_step_id

	return