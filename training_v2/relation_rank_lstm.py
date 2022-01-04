import argparse
import copy
import numpy as np
import os

# import psutil
import random
import tensorflow as tf
from time import time

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops

    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)


from load_data import load_EOD_data, load_relation_data
from evaluator import evaluate
from utils import add_custom_summary


tf.compat.v1.disable_eager_execution()

seed = 123456789
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
CHECKPOINT_DIR = ".checkpoints"
MODEL_OUTPUT_DIR = "model_outputs"
LOG_DIR = "logs"


class ReRaLSTM:
    def __init__(
        self,
        data_path,
        market_name,
        tickers_fname,
        relation_name,
        emb_fname,
        parameters,
        steps=1,
        epochs=50,
        batch_size=None,
        flat=False,
        gpu=False,
        in_pro=False,
    ):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(
            os.path.join(data_path, "..", tickers_fname),
            dtype=str,
            delimiter="\t",
            skip_header=False,
        )

        print("#tickers selected:", len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = load_EOD_data(
            data_path, market_name, self.tickers, steps
        )

        # relation data
        rname_tail = {
            "sector_industry": "_industry_relation.npy",
            "wikidata": "_wiki_relation.npy",
        }

        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(
                self.data_path,
                "..",
                "relation",
                self.relation_name,
                self.market_name + rname_tail[self.relation_name],
            )
        )
        print("relation encoding shape:", self.rel_encoding.shape)
        print("relation mask shape:", self.rel_mask.shape)

        self.embedding = np.load(
            os.path.join(self.data_path, "..", "pretrain", emb_fname)
        )
        print("embedding shape:", self.embedding.shape)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters["seq"]
        mask_batch = self.mask_data[:, offset : offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            self.embedding[:, offset, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1),
        )

    def train(self):
        if self.gpu == True:
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        print("device name:", device_name)
        with tf.device(device_name):
            tf.compat.v1.reset_default_graph()

            seed = 123456789
            random.seed(seed)
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)

            ground_truth = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, 1]
            )  # N*1 (when batch_size=None)
            mask = tf.compat.v1.placeholder(tf.float32, [self.batch_size, 1])  # N*1
            feature = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, self.parameters["unit"]]
            )  # N*U
            base_price = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, 1]
            )  # N*1
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)  # N*1

            relation = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(self.batch_size, self.batch_size, 108)
            )  # N * N * K
            rel_mask = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(self.batch_size, self.batch_size)
            )  # N * N

            rel_weight = tf.compat.v1.layers.dense(
                relation, units=1, activation=leaky_relu
            )  # N*N*1 (create weight of shape K*1 and operate on last index of relation (K))

            if self.inner_prod:
                print("inner product weight")
                inner_weight = tf.matmul(feature, feature, transpose_b=True)  # N*N
                weight = tf.multiply(inner_weight, rel_weight[:, :, -1])  # N*N
            else:
                print("sum weight")
                head_weight = tf.compat.v1.layers.dense(
                    feature, units=1, activation=leaky_relu
                )  # N*1
                tail_weight = tf.compat.v1.layers.dense(
                    feature, units=1, activation=leaky_relu
                )  # N*1
                weight = tf.add(
                    tf.add(
                        tf.matmul(head_weight, all_one, transpose_b=True),  # N*N
                        tf.matmul(all_one, tail_weight, transpose_b=True),  # N*N
                    ),
                    rel_weight[:, :, -1],
                )  # N*N
            weight_masked = tf.nn.softmax(tf.add(rel_mask, weight), axis=0)
            outputs_proped = tf.matmul(weight_masked, feature)

            if self.flat:
                print("one more hidden layer")
                outputs_concated = tf.compat.v1.layers.dense(
                    tf.concat([feature, outputs_proped], axis=1),
                    units=self.parameters["unit"],
                    activation=leaky_relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                )
            else:
                outputs_concated = tf.concat([feature, outputs_proped], axis=1)

            # One hidden layer
            prediction = tf.compat.v1.layers.dense(
                outputs_concated,
                units=1,
                activation=leaky_relu,
                name="reg_fc",
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            )

            return_ratio = tf.compat.v1.div(
                tf.subtract(prediction, base_price), base_price
            )
            reg_loss = tf.compat.v1.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio, transpose_b=True),
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth, transpose_b=True),
                tf.matmul(ground_truth, all_one, transpose_b=True),
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)
            rank_loss = tf.reduce_mean(
                input_tensor=tf.nn.relu(
                    tf.multiply(tf.multiply(pre_pw_dif, gt_pw_dif), mask_pw)
                )
            )
            loss = reg_loss + tf.cast(self.parameters["alpha"], tf.float32) * rank_loss
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.parameters["lr"]
            ).minimize(loss)
        # TF2
        # checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=optimizer)
        # latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        # if latest_checkpoint:
        #     status = checkpoint.restore(latest_checkpoint)
        #     status.assert_consumed()
        #     print("Restored from {}".format(latest_checkpoint))
        # else:
        #     print("Initializing from scratch.")
        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()

        # Not working with current tensorboard, possibly due to large const size of "relation" layer
        train_writer = tf.compat.v1.summary.FileWriter(
            LOG_DIR + "/train", graph=sess.graph
        )
        val_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + "/validation")
        test_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + "/test")
        sess.run(tf.compat.v1.global_variables_initializer())

        # restore checkpoints
        # TF2
        # checkpoint = tf.train.Checkpoint(
        #     var_list={v.name.split(":")[0]: v for v in tf.compat.v1.global_variables()}
        # )
        # latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        # TF1
        latest_checkpoint = tf.compat.v1.train.latest_checkpoint(f"{CHECKPOINT_DIR}")
        latest_checkpoint_num = (
            int(latest_checkpoint.split("-")[-1]) if latest_checkpoint else 0
        )
        if latest_checkpoint:
            print(f"Lastest checkpoint: {latest_checkpoint}")
            saver.restore(sess=sess, save_path=latest_checkpoint)
            print("Restored from {}".format(latest_checkpoint))
        else:
            print("Initializing from scratch.")

        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index], dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index], dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index], dtype=float
        )
        best_test_pred = np.zeros(
            [
                len(self.tickers),
                self.trade_dates
                - self.parameters["seq"]
                - self.test_index
                - self.steps
                + 1,
            ],
            dtype=float,
        )
        best_test_gt = np.zeros(
            [
                len(self.tickers),
                self.trade_dates
                - self.parameters["seq"]
                - self.test_index
                - self.steps
                + 1,
            ],
            dtype=float,
        )
        best_test_mask = np.zeros(
            [
                len(self.tickers),
                self.trade_dates
                - self.parameters["seq"]
                - self.test_index
                - self.steps
                + 1,
            ],
            dtype=float,
        )
        best_valid_perf = {"mse": np.inf, "mrrt": 0.0, "btl": 0.0}
        best_test_perf = {"mse": np.inf, "mrrt": 0.0, "btl": 0.0}
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            global_step = latest_checkpoint_num + i + 1
            for j in range(self.valid_index - self.parameters["seq"] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j]
                )
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch,
                    relation: self.rel_encoding,
                    rel_mask: self.rel_mask,
                }
                cur_loss, cur_reg_loss, cur_rank_loss, batch_out = sess.run(
                    (loss, reg_loss, rank_loss, optimizer), feed_dict
                )
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
            print(
                "Train Loss:",
                tra_loss / (self.valid_index - self.parameters["seq"] - self.steps + 1),
                tra_reg_loss
                / (self.valid_index - self.parameters["seq"] - self.steps + 1),
                tra_rank_loss
                / (self.valid_index - self.parameters["seq"] - self.steps + 1),
            )
            train_writer.add_summary(
                add_custom_summary(
                    "tra_loss",
                    tra_loss
                    / (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ),
                global_step=global_step,
            )
            train_writer.add_summary(
                add_custom_summary(
                    "tra_reg_loss",
                    tra_reg_loss
                    / (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ),
                global_step=global_step,
            )
            train_writer.add_summary(
                add_custom_summary(
                    "tra_rank_loss",
                    tra_rank_loss
                    / (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ),
                global_step=global_step,
            )
            # save checkpoints
            # TF2
            # checkpoint.save(CHECKPOINT_DIR)
            # ckpt = tf.train.Checkpoint(
            #     var_list={v.name.split(':')[0]: v for v in tf.compat.v1.global_variables()})

            # TF1
            saver.save(
                sess,
                f"{CHECKPOINT_DIR}/{self.market_name}",
                global_step=global_step,
            )
            print(
                f"Saved check point: {CHECKPOINT_DIR}/{self.market_name}-{global_step}"
            )

            # test on validation set
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(
                self.valid_index - self.parameters["seq"] - self.steps + 1,
                self.test_index - self.parameters["seq"] - self.steps + 1,
            ):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset
                )
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch,
                    relation: self.rel_encoding,
                    rel_mask: self.rel_mask,
                }
                (
                    cur_loss,
                    cur_reg_loss,
                    cur_rank_loss,
                    cur_rr,
                ) = sess.run((loss, reg_loss, rank_loss, return_ratio), feed_dict)
                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[
                    :,
                    cur_offset
                    - (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(cur_rr[:, 0])
                cur_valid_gt[
                    :,
                    cur_offset
                    - (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(gt_batch[:, 0])
                cur_valid_mask[
                    :,
                    cur_offset
                    - (self.valid_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(mask_batch[:, 0])
            print(
                "Valid MSE:",
                val_loss / (self.test_index - self.valid_index),
                val_reg_loss / (self.test_index - self.valid_index),
                val_rank_loss / (self.test_index - self.valid_index),
            )
            val_writer.add_summary(
                add_custom_summary(
                    "val_loss", val_loss / (self.test_index - self.valid_index)
                ),
                global_step=global_step,
            )
            val_writer.add_summary(
                add_custom_summary(
                    "val_reg_loss", val_reg_loss / (self.test_index - self.valid_index)
                ),
                global_step=global_step,
            )
            val_writer.add_summary(
                add_custom_summary(
                    "val_rank_loss",
                    val_rank_loss / (self.test_index - self.valid_index),
                ),
                global_step=global_step,
            )
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            print("\t Valid preformance:", cur_valid_perf)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(
                self.test_index - self.parameters["seq"] - self.steps + 1,
                self.trade_dates - self.parameters["seq"] - self.steps + 1,
            ):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset
                )
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch,
                    relation: self.rel_encoding,
                    rel_mask: self.rel_mask,
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = sess.run(
                    (loss, reg_loss, rank_loss, return_ratio), feed_dict
                )
                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

                cur_test_pred[
                    :,
                    cur_offset
                    - (self.test_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(cur_rr[:, 0])
                cur_test_gt[
                    :,
                    cur_offset
                    - (self.test_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(gt_batch[:, 0])
                cur_test_mask[
                    :,
                    cur_offset
                    - (self.test_index - self.parameters["seq"] - self.steps + 1),
                ] = copy.copy(mask_batch[:, 0])
            print(
                "Test MSE:",
                test_loss / (self.trade_dates - self.test_index),
                test_reg_loss / (self.trade_dates - self.test_index),
                test_rank_loss / (self.trade_dates - self.test_index),
            )
            test_writer.add_summary(
                add_custom_summary(
                    "test_loss", test_loss / (self.trade_dates - self.test_index)
                ),
                global_step=global_step,
            )
            test_writer.add_summary(
                add_custom_summary(
                    "test_reg_loss",
                    test_reg_loss / (self.trade_dates - self.test_index),
                ),
                global_step=global_step,
            )
            test_writer.add_summary(
                add_custom_summary(
                    "test_rank_loss",
                    test_rank_loss / (self.trade_dates - self.test_index),
                ),
                global_step=global_step,
            )
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print("\t Test performance:", cur_test_perf)
            if val_loss / (self.test_index - self.valid_index) < best_valid_loss:
                best_valid_loss = val_loss / (self.test_index - self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)
                print("Better valid loss:", best_valid_loss)
            t4 = time()
            print("epoch:", i, ("time: %.4f " % (t4 - t1)))
        print("\nBest Valid performance:", best_valid_perf)
        print("\tBest Test performance:", best_test_perf)

        builder = tf.compat.v1.saved_model.Builder(
            f"{MODEL_OUTPUT_DIR}/{self.market_name}/{latest_checkpoint_num}"
        )
        builder.add_meta_graph_and_variables(sess, tags=["serve"])
        builder.save(as_text=False)
        sess.close()

        tf.compat.v1.reset_default_graph()
        return (
            best_valid_pred,
            best_valid_gt,
            best_valid_mask,
            best_test_pred,
            best_test_gt,
            best_test_mask,
        )

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == "__main__":
    desc = "train a relational rank lstm model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-p", help="path of EOD data", default="../data/2013-01-01")
    parser.add_argument("-m", help="market name", default="NASDAQ")
    parser.add_argument("-t", help="fname for selected tickers")
    parser.add_argument(
        "-l", default=4, help="length of historical sequence for feature"
    )
    parser.add_argument("-u", default=64, help="number of hidden units in lstm")
    parser.add_argument("-s", default=1, help="steps to make prediction")
    parser.add_argument("-r", default=0.001, help="learning rate")
    parser.add_argument("-a", default=1, help="alpha, the weight of ranking loss")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="use gpu")

    parser.add_argument(
        "-e",
        "--emb_file",
        type=str,
        default="NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy",
        help="fname for pretrained sequential embedding",
    )
    parser.add_argument(
        "-rn",
        "--rel_name",
        type=str,
        default="sector_industry",
        help="relation type: sector_industry or wikidata",
    )
    parser.add_argument("-ip", "--inner_prod", type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + "_tickers_qualify_dr-0.98_min-5_smooth.csv"
    args.gpu = args.gpu == 1

    args.inner_prod = args.inner_prod == 1

    parameters = {
        "seq": int(args.l),
        "unit": int(args.u),
        "lr": float(args.r),
        "alpha": float(args.a),
    }
    print("arguments:", args)
    print("parameters:", parameters)

    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1,
        epochs=2,
        batch_size=None,
        gpu=args.gpu,
        in_pro=args.inner_prod,
    )

    pred_all = RR_LSTM.train()
