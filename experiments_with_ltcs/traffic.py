import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Run on CPU

import tensorflow as tf

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse
import datetime as dt


def load_trace():
    df = pd.read_csv("data/traffic/Metro_Interstate_Traffic_Volume.csv")
    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    date_time = df["date_time"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)  # normalize
    traffic_volume /= np.std(traffic_volume)  # normalize

    return features, traffic_volume


def load_bikes(predict_hour=1):
    df = pd.read_csv("data/bicikelj/bicikelj_one.csv")
    #holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp_c"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["will_it_rain"].values.astype(np.float32)
    #snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["cloud"].values.astype(np.float32)
    date_time = df["date"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    hour_of_day = np.array([d.hour for d in date_time]).astype(np.float32)
    day_of_week = np.array([d.weekday() for d in date_time]).astype(np.float32)

    bikes_plus_120mins = df["bikes_plus_120mins"].values.astype(np.float32)
    bikes_plus_60mins = df["bikes_plus_60mins"].values.astype(np.float32)


    traffic_volume = df["value"].values.astype(np.float32)

    prev_hour_bikes = np.zeros_like(traffic_volume)
    prev_hour_bikes[1:] = traffic_volume[:-1]
    prev_hour_bikes[0] = traffic_volume[0]



    features = np.stack([temp, rain, clouds, weekday, hour_of_day,day_of_week,prev_hour_bikes,traffic_volume], axis=-1)

    # traffic_volume -= np.mean(traffic_volume)  # normalize
    # traffic_volume /= np.std(traffic_volume)  # normalize

    #return date_time,features, traffic_volume
    if predict_hour == 1:
        return date_time,features, bikes_plus_60mins

    elif predict_hour == 2:
        return date_time,features, bikes_plus_120mins


def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

import numpy as np

class TrafficData:
    def __init__(self, seq_len=32, exclude_timestamps=None,predict_hour=1):

        date_time, x, y = load_bikes(predict_hour=predict_hour)
        date_time = np.array(date_time)
        x = np.array(x)
        y = np.array(y)

        if exclude_timestamps is not None:
            # Convert exclude_timestamps to datetime objects
            exclude_timestamps = [dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in exclude_timestamps]

            # Filter out the timestamps to be excluded
            indices_to_keep = [i for i, dt in enumerate(date_time) if dt not in exclude_timestamps]
            date_time = date_time[indices_to_keep]
            x = x[indices_to_keep]
            y = y[indices_to_keep]

        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=1)

        print("train_x.shape: {}".format(train_x.shape))
        print("train_y.shape: {}".format(train_y.shape))

        print("train_x[0]: {}".format(train_x[0]))
        print("train_y[0]: {}".format(train_y[0]))


        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.2 * total_seqs)
        test_size = int(0.3 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]

    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)



class TrafficModel:
    def __init__(self, model_type, model_size, learning_rate=0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 8])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.model_size = model_size
        head = self.x
        if model_type == "lstm":
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type.startswith("ltc"):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if model_type.endswith("_rk"):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif model_type.endswith("_ex"):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(
                self.wm, head, dtype=tf.float32, time_major=True
            )
            self.constrain_op = self.wm.get_param_constrain_op()
        elif model_type == "node":
            self.fused_cell = NODE(model_size, cell_clip=10)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctgru":
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctrnn":
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        target_y = tf.expand_dims(self.target_y, axis=-1)
        self.y = tf.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join(
            "results", "traffic", "{}_{}.csv".format(model_type, model_size)
        )
        if not os.path.exists("results/traffic"):
            os.makedirs("results/traffic")
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n"
                )

        self.checkpoint_path = os.path.join(
            "tf_sessions", "traffic", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/traffic"):
            os.makedirs("tf_sessions/traffic")

        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, gesture_data, epochs, verbose=True, log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.test_x, self.target_y: gesture_data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.valid_x, self.target_y: gesture_data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x, batch_y in gesture_data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y},
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))):
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        with open(self.result_file, "a") as f:
            f.write(
                "{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                    best_epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )

def predict_bikes(timestamp,predict_hour=1):
    # Convert timestamp to datetime object
    timestamp = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    # Load the dataset
    date_time,features, traffic_volume = load_bikes(predict_hour=predict_hour)

    # Convert the dates in the dataset to datetime objects
   

    # Find the index of the closest hour that is not the current hour
    diff = [abs((d - timestamp).total_seconds()) for d in date_time]
    closest_index = np.argmin(diff)
    if date_time[closest_index].hour == timestamp.hour:
        diff[closest_index] = np.inf
        closest_index = np.argmin(diff)
        

    # Compute the features for the closest hour
    closest_features = features[closest_index]
    # print(closest_features)
    # print("closest time: ", date_time[closest_index])

    # Reshape the features to match the input shape of the model
    closest_features = np.reshape(closest_features, (1, 1, -1))

    # Pass the features through the model to make a prediction
    prediction = model.sess.run(model.y, {model.x: closest_features})


    return prediction[0, 0, 0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--log", default=1, type=int)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    timestamps = [
    "2022-08-04 23:24:00",
    "2022-08-05 00:24:00",
    "2022-08-08 04:58:00",
    "2022-08-08 05:58:00",
    "2022-08-11 21:13:00",
    "2022-08-11 22:13:00",
    "2022-08-16 07:10:00",
    "2022-08-16 08:10:00",
    "2022-08-18 17:30:00",
    "2022-08-18 18:27:00",
    "2022-08-21 11:07:00",
    "2022-08-21 12:07:00",
    "2022-08-24 15:09:00",
    "2022-08-24 16:09:00",
    "2022-08-29 03:05:00",
    "2022-08-29 04:01:00",
    "2022-08-31 06:37:00",
    "2022-08-31 07:34:00",
    "2022-09-03 15:48:00",
    "2022-09-03 16:42:00",
    "2022-09-05 18:34:00",
    "2022-09-05 19:35:00",
    "2022-09-08 07:44:00",
    "2022-09-08 08:44:00",
    "2022-09-10 05:50:00",
    "2022-09-10 06:50:00",
    "2022-09-12 09:32:00",
    "2022-09-12 10:32:00",
    "2022-09-17 18:25:00",
    "2022-09-17 19:25:00",
    "2022-09-20 23:43:00",
    "2022-09-21 00:40:00",
    "2022-09-23 13:36:00",
    "2022-09-23 14:36:00",
    "2022-09-26 07:09:00",
    "2022-09-26 08:10:00",
    "2022-09-29 03:14:00",
    "2022-09-29 04:10:00",
    "2022-10-01 19:27:00",
    "2022-10-01 20:27:00",
    ]

    traffic_data = TrafficData(exclude_timestamps=timestamps,predict_hour=1)
    model = TrafficModel(model_type=args.model, model_size=args.size)

    model.fit(traffic_data, epochs=args.epochs, log_period=args.log)

    model.restore()


    predictions = []

  
    i = 0
    for timestamp in timestamps:
        # print("Predicting bikes for {}".format(timestamp))

        if i % 2 == 0:
            prediction = predict_bikes(timestamp,predict_hour=1)
            predictions.append(prediction)


        i += 1

    print(predictions)

    #reset the graph
    tf.reset_default_graph()

    traffic_data = TrafficData(exclude_timestamps=timestamps,predict_hour=2)
    model = TrafficModel(model_type=args.model, model_size=args.size)
    model.fit(traffic_data, epochs=args.epochs, log_period=args.log)
    model.restore()

    i = 0
    for timestamp in timestamps:
        # print("Predicting bikes for {}".format(timestamp))

        if i % 2 == 0:
           pass
        else:
            prediction = predict_bikes(timestamp,predict_hour=2)
            predictions.append(prediction)

        i += 1


    print(predictions)

    #save the predictions to txt
    with open("predictions.txt", "w") as f:

        for prediction in predictions:
            f.write("{:0.8f}\n".format(prediction))

    


    




    


