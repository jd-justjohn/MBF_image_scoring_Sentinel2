import tensorflow as tf
from sklearn.metrics import r2_score
import normalizing_functions
from functools import partial
import numpy as np
import os
from data_loader import get_dataset


##################################################### data loader #########################################################
crop = ['corn', 'soy', 'winter_wheat']

lowest_dirs_corn = list()
top_level_dir_path = '/home/ubuntu/Downloads/corn_hexes_and_imgery_latest_small_files'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_corn.append(root)

lowest_dirs_soy = list()
top_level_dir_path = '/home/ubuntu/Downloads/soy_hexes_and_imgery_latest_small_files'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_soy.append(root)

lowest_dirs_wheat = list()
top_level_dir_path = '/home/ubuntu/Downloads/wheat_hexes_and_imagery'
for root,dirs,files in os.walk(top_level_dir_path):
    if not dirs:
        lowest_dirs_wheat.append(root)

lowest_dirs_list = [lowest_dirs_corn, lowest_dirs_soy, lowest_dirs_wheat]

val_files = [[x for x in yy if 'seeding_date_min_by_fop=2021' in x] for yy in lowest_dirs_list]

num_bands=12
shuffle_buffer=200000
batch_size = 35
crop_season_only = True
predict_on_N_imgs = 20
perc_field_fill=0.8
min_num_hexes_per_field=50
parametric=False

val_ds_list = [get_dataset(yy,shuffle_buffer=2000000,batch_size=batch_size, crop_season_only=crop_season_only, predict_on_N_imgs=100,
                     num_bands=num_bands,perc_field_fill=perc_field_fill, min_num_hexes_per_field=min_num_hexes_per_field, parametric=parametric).prefetch(buffer_size=tf.data.AUTOTUNE) for yy in val_files]
val_ds = tf.data.Dataset.zip(tuple(val_ds_list))
########################################################################################################################

best_top_num = 3
model = tf.keras.models.load_model('/home/ubuntu/Documents/keras_model_training/tensorboard/regression/best_img_r2/20230102-021200/my_model_best_ranking')

def prep_batch(x_in, y_in):
    y_in = tf.reshape(y_in, shape=[-1, 1])
    x_in = tf.reshape(x_in, shape=[-1, *tf.shape(x_in)[2:]])
    y_not_zero = tf.not_equal(y_in, -1)
    y_in = tf.gather(y_in, tf.where(y_not_zero)[:, 0], axis=0).numpy()
    x_in = tf.gather(x_in, tf.where(y_not_zero)[:, 0], axis=0).numpy()
    y_in = y_in[np.all(x_in > 0, axis=1), :]
    x_in = x_in[np.all(x_in > 0, axis=1), :]
    x_in, y = normalizing_functions.normalize(x_in, y_in, 'all_crops')
    return x_in, tf.convert_to_tensor(y_in)

tt_ = val_ds.as_numpy_iterator().next()

def ranking_performance(idx_inner, idx_outer):
    x_sub, y_sub, z_sub = tt_[idx_outer][0][idx_inner:idx_inner+1], tt_[idx_outer][1][idx_inner:idx_inner+1], tt_[idx_outer][2][idx_inner]
    x_sub, y_sub = prep_batch(x_sub, y_sub)

    y_pred = model(x_sub)
    y_sub_sorted = tf.gather(y_sub, tf.argsort(tf.reshape(y_pred, [-1])))
    random_indxs = tf.random.uniform((best_top_num,), maxval=len(y_sub), dtype=tf.int32)
    return tf.reduce_max(y_sub_sorted[-tf.minimum(len(y_sub_sorted), best_top_num):])/z_sub, tf.reduce_max(tf.gather(y_sub, random_indxs)) / z_sub

best_top_list = []
best_top_rand_samp_list = []
for i in range(len(tt_)):
    part_ranking_performance = partial(ranking_performance, idx_outer=i)
    best_top, best_top_rand_samp = tf.map_fn(part_ranking_performance, tf.range(len(tt_[i][-1])), fn_output_signature=(tf.float32, tf.float32))
    best_top_list.append(best_top)
    best_top_rand_samp_list.append(best_top_rand_samp)

batch_prepped = [prep_batch(xx[0], xx[1]) for xx in tt_]
b_lens = np.cumsum([0] + [len(xx[0]) for xx in batch_prepped])
x = tf.concat([xx[0] for xx in batch_prepped], axis=0)
y = tf.concat([xx[1] for xx in batch_prepped], axis=0)

y_pred = model(x) ## input is quantiles of each of 12 bands for [5, 25, 50, 75, 95].  Order is quantiles (e.g. 5% quantiles, then 25% quantiles for all bands)

r2_score_val = r2_score(y.numpy(), y_pred.numpy())
best_top = [np.mean(best_top_list[i].numpy()) for i in range(len(best_top_list))] #same order as "crop" list
best_top_random = [np.mean(best_top_rand_samp_list[i].numpy()) for i in range(len(best_top_rand_samp_list))] #same order as "crop" list