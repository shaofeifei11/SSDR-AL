from os import makedirs
import time
from os import makedirs
from os.path import exists
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import helper_tf_util
from s3dis_dataset import *
from s3dis_dataset_test import *
from helper_ply import write_ply

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

early_stop_count = 6

class  Network:
    def __init__(self, config, dataset_name, sampler_args, test_area_idx, reg_strength):
        self.config = config
        self.dataset_name = dataset_name
        self.sampler_args = sampler_args
        self.test_area_idx = test_area_idx
        self.reg_strength = reg_strength

        self.Log_file = open(join("record_log", 'log_train_' + dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) +"_"+str(reg_strength)+ '.txt'), 'a')
        self.init_input()
        self.training_epoch = 0
        self.correct_prediction = 0
        self.accuracy = 0
        self.class_weights = DP.get_class_weights(dataset_name)


        with tf.variable_scope('layers', reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.logits_3d, self.last_second_features = self.inference(self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.logits = tf.reshape(self.logits_3d, [-1, config.num_classes])
            self.last_second_features = tf.reshape(self.last_second_features, [-1, 32])

            self.labels = tf.reshape(self.input_labels, [-1])
            self.activation = tf.reshape(self.input_activation, [-1])
            self.pseudo = tf.reshape(self.input_pseudo, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx_init = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_idx = tf.reshape(valid_idx_init, [-1])

            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_activation = tf.gather(self.activation, valid_idx, axis=0)

            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)
            valid_pseudo_init = tf.gather(self.pseudo, valid_idx, axis=0)
            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)
            valid_pseudo = tf.gather(reducing_list, valid_pseudo_init)


            self.loss = self.get_loss(valid_logits, valid_pseudo, valid_activation, self.class_weights)

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results', reuse=tf.AUTO_REUSE):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        self.saving_path = join("./data", dataset_name, str(reg_strength), "saver", get_sampler_args_str(self.sampler_args), "snapshots")
        makedirs(self.saving_path) if not exists(self.saving_path) else None
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()

        self.tensorboard_path = join("./data", dataset_name, str(reg_strength), "saver", get_sampler_args_str(self.sampler_args), "tensorboard")
        makedirs(self.tensorboard_path) if not exists(self.tensorboard_path) else None

        self.train_writer = tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def restore_model(self, round_num):
        if round_num == 1:

            restore_snap = join("./data", self.dataset_name, str(self.reg_strength), "saver", "seed", "snapshots", 'snap-{:d}'.format(1))
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from seed")

        # Load trained model
        else:
            restore_snap = join(self.saving_path, 'snap-{:d}'.format(round_num))
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

    def restore_baseline_model(self):

        restore_snap = join("./data", self.dataset_name, str(self.reg_strength), "saver", "baseline", "snapshots", 'snap-{:d}'.format(1))
        self.saver.restore(self.sess, restore_snap)
        print("Model restored from baseline")

    def init_input(self):
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            self.input_xyz, self.input_neigh_idx, self.input_sub_idx, self.input_interp_idx = [], [], [], []
            for i in range(self.config.num_layers):
                self.input_xyz.append(tf.placeholder(tf.float32, shape=[None, None, 3]))  #[batch, point, 3]
                self.input_neigh_idx.append(tf.placeholder(tf.int32, shape=[None, None, self.config.k_n]))  #[batch, point, 16]
                self.input_sub_idx.append(tf.placeholder(tf.int32, shape=[None, None, self.config.k_n]))  #[batch, point, 16]
                self.input_interp_idx.append(tf.placeholder(tf.int32, shape=[None, None, 1]))  #[batch, point, 3]
            self.input_features = tf.placeholder(tf.float32, shape=[None, None, 6])  #[batch, point, 3+3]
            self.input_labels = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_activation = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_pseudo = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_input_inds = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_cloud_inds = tf.placeholder(tf.int32, shape=[None])  # [batch]

    def inference(self, is_training):

        d_out = self.config.d_out
        feature = self.input_features
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)


        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, self.input_xyz[i], self.input_neigh_idx[i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, self.input_sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)


        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, self.input_interp_idx[-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, f_layer_fc2

    def get_feed_dict(self, dat, is_training):
        feed_dict = {self.is_training: is_training}
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = np.squeeze(dat[j].numpy(), axis=1)  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = np.squeeze(dat[self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = np.squeeze(dat[2 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = np.squeeze(dat[3 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 3]


        feed_dict[self.input_features] = np.squeeze(dat[4 * self.config.num_layers+0].numpy(), axis=1)  # [batch, point, 3+3]
        feed_dict[self.input_labels] = np.squeeze(dat[4 * self.config.num_layers+1].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_activation] = np.squeeze(dat[4 * self.config.num_layers+2].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_pseudo] = np.squeeze(dat[4 * self.config.num_layers+3].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_input_inds] = np.squeeze(dat[4 * self.config.num_layers+4].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_cloud_inds] = np.squeeze(dat[4 * self.config.num_layers+5].numpy(), axis=1)  # [batch]

        return feed_dict

    def get_feed_dict_test(self, dat):
        feed_dict = {self.is_training: False}
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = dat[j]  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = dat[self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = dat[2 * self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = dat[3 * self.config.num_layers + j]  # [batch, point, 3]
        feed_dict[self.input_features] = dat[4 * self.config.num_layers+0]  # [batch, point, 3+3]
        feed_dict[self.input_labels] = dat[4 * self.config.num_layers+1]  # [batch, point]
        feed_dict[self.input_input_inds] = dat[4 * self.config.num_layers+2]  # [batch, point]
        feed_dict[self.input_cloud_inds] = dat[4 * self.config.num_layers+3]  # [batch]
        return feed_dict

    def reset_lr(self):
        op = self.learning_rate.assign(self.config.learning_rate)
        self.sess.run(op)

    def train(self, round_num):
        self.reset_lr()
        self.training_epoch = 0
        log_out("Round "+str(round_num) + ' | ****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        best_miou = 0
        best_OA = 0
        train_data, train_loader, val_data, val_loader, test_data, test_probs = None, None, None, None, None, None
        if self.dataset_name == "S3DIS":
            train_data = S3DIS_Dataset(test_area_idx=self.test_area_idx, sampler_args=self.sampler_args, round_num=round_num, mode="training", reg_strength=self.reg_strength)
            train_loader = DataLoader(train_data, batch_size=6, shuffle=True, num_workers=6)
            test_data = S3DIS_Dataset_Test(self.test_area_idx)
            test_probs = [np.zeros(shape=[l.shape[0], self.config.num_classes], dtype=np.float32) for l in test_data.input_labels]

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            one_epoch_steps = int(self.config.batch_size*self.config.train_steps/len(train_data) + 1)
            activation_sum = 0
            for step in range(one_epoch_steps):
                summary, l_out, probs, labels, acti, acc = None, None, None, None, None, None
                activation_sum = 0
                for i, dat in enumerate(train_loader):
                    ops = [self.train_op,
                               self.extra_update_ops,
                               self.merged,
                               self.loss,
                               self.logits,
                               self.labels,
                               self.activation,
                               self.accuracy]
                    _, _, summary, l_out, probs, labels, acti, acc = self.sess.run(ops, feed_dict=self.get_feed_dict(dat, True))

                    activation_sum = activation_sum + np.sum(acti)

                self.train_writer.add_summary(summary, self.training_epoch*one_epoch_steps+step)
                t_end = time.time()
                message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                log_out("Round "+str(round_num) + ' | epoch=' + str(self.training_epoch) + ' | '+ message.format(step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)

            log_out("Round "+str(round_num) + ' | epoch=' + str(self.training_epoch) + ", train costTime=" + str(time.time()-t_start) + ", | total_activation_sum=" + str(activation_sum), self.Log_file)
            self.training_epoch += 1
            # Update learning rate
            op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                       self.config.lr_decays[self.training_epoch]))
            self.sess.run(op)

            if self.training_epoch >= int(self.config.max_epoch * 0.4):
                tt12 = time.time()
                if self.dataset_name in ["S3DIS", "SemanticKITTI"] :
                    if self.dataset_name == "S3DIS":
                        m_iou, OA = self.evaluate_test_s3dis(dataset=test_data, test_probs=test_probs)
                    if m_iou > best_miou:
                        # Save the best model
                        snapshot_directory = join(self.saving_path)
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, join(self.saving_path, "snap"), global_step=round_num)
                        early_count = 0
                        best_miou = m_iou
                        best_OA = OA

                log_out("Round " + str(round_num) + ' | Best m_IoU is: {:5.3f}'.format(
                    best_miou) + ', OA is: {:5.3f}'.format(best_OA) +
                        " | val costTime=" + str(time.time() - tt12), self.Log_file)

            log_out("Round "+str(round_num) + ' | ****EPOCH {}****'.format(self.training_epoch), self.Log_file)

        return best_miou, best_OA

    def close(self):
        print('finished')
        self.train_writer.close()
        self.sess.close()
        self.Log_file.close()

    def evaluate_test_s3dis(self, dataset, test_probs):
        num_votes = 100

        dataset.init_possibility()
        # Smoothing parameter for votes
        test_smooth = 0.95

        # Number of points per class in validation set
        val_proportions = np.zeros(self.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        m_IoU, OA = 0, 0

        while last_min < num_votes:
            dat = dataset.get_batch()
            while len(dat) > 0:
                ops = (self.prob_logits,
                       self.labels,
                       self.input_input_inds,
                       self.input_cloud_inds,
                       )

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops,
                                                                                    feed_dict=self.get_feed_dict_test(
                                                                                        dat))

                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                # print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs,
                                           [self.config.val_batch_size, self.config.num_points,
                                            self.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j]
                    test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

                dat = dataset.get_batch()

            new_min = np.min(dataset.min_possibility)
            # log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                # log_out('\nConfusion on sub clouds', self.Log_file)
                confusion_list = []

                num_val = len(dataset.input_labels)

                for i_test in range(num_val):
                    probs = test_probs[i_test]
                    preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                    labels = dataset.input_labels[i_test]

                    # Confs
                    confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                # Rescale with the right number of point per class
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # Compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                # log_out(s + '\n', self.Log_file)

                if int(np.ceil(new_min)) % 1 == 0:

                    # Project predictions
                    # log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                    proj_probs_list = []

                    for i_val in range(num_val):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.val_proj[i_val]
                        probs = test_probs[i_val][proj_idx, :]
                        proj_probs_list += [probs]

                    # Show vote results
                    # log_out('Confusion on full clouds', self.Log_file)
                    val_total_correct = 0
                    val_total_seen = 0
                    confusion_list = []
                    for i_test in range(num_val):
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

                        # Confusion
                        labels = dataset.val_labels[i_test]
                        correct = np.sum(preds == labels)
                        val_total_correct += correct
                        val_total_seen += len(labels)
                        # log_out(dataset.input_names[i_test] + ' Acc:' + str(acc), self.Log_file)

                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0)

                    OA = val_total_correct / float(val_total_seen)

                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    # log_out('-' * len(s), self.Log_file)
                    # log_out(s, self.Log_file)
                    # log_out('-' * len(s) + '\n', self.Log_file)
                    print('finished \n')
                    return m_IoU, OA

            epoch_id += 1
            step_id = 0
            continue
        return m_IoU, OA

    def evaluate(self, val_data, val_loader, round_num):
        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(int(self.config.val_batch_size*self.config.val_steps/len(val_data))+1):
        # for step_id in range(self.config.val_steps):
            for i, dat in enumerate(val_loader):
                ops = (self.prob_logits, self.labels, self.accuracy)

                stacked_prob, labels, acc = self.sess.run(ops, feed_dict=self.get_feed_dict(dat, False))
                pred = np.argmax(stacked_prob, 1)

                if not self.config.ignored_label_inds or len(self.config.ignored_label_inds) == 0:
                    pred_valid = pred
                    labels_valid = labels
                else:

                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

        iou_list = []
        for n in range(0, self.config.num_classes, 1):

            union = float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            if union == 0.0:
                union = 1.0
            iou = true_positive_classes[n] / union

            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)
        OA = val_total_correct / float(val_total_seen)

        log_out("Round "+str(round_num) + ' | OA: {}'.format(OA), self.Log_file)
        log_out("Round "+str(round_num) + ' | mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out("Round "+str(round_num) + ' | Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out("Round "+str(round_num) + ' | -' * len(s), self.Log_file)
        log_out("Round "+str(round_num) + ' | '+ s, self.Log_file)
        log_out("Round "+str(round_num) + ' | -' * len(s) + '\n', self.Log_file)

        return mean_iou, OA

    def get_loss(self, logits, labels, activation, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)


        self.ssdr_one_hot_labels = one_hot_labels
        self.ssdr_logits = logits


        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)

        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)

        weighted_losses = unweighted_losses * weights
        weighted_losses_acti = weighted_losses * tf.cast(activation, dtype=tf.float32)
        output_loss = tf.reduce_mean(weighted_losses_acti)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):

        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):

        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):

        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg

if __name__=="__main__":
    b = tf.ones()