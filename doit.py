# %% [markdown]
# First we have to download and install the pip package

# %%
import numpy as np
import torch.nn as nn
from ncps.wirings import AutoNCP 
from ncps.torch import LTC
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd








# %%


# %%
import datetime as dt
def load_bikes(predict_hour=1,dataset_file="/d/hpc/home/ld8435/liquid_time_constant_networks/experiments_with_ltcs/data/bicikelj/bicikelj_one.csv"):
    df = pd.read_csv(dataset_file)

    #resample to 10 minutes
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.resample("1h").mean()
    df = df.reset_index()

    

    #back to string
    df["timestamp"] = df["timestamp"].astype(str)

    #set missing values to 0

    #remove missing values
    df = df.dropna()

    #fill temp with averagte temp
    df["temp_c"] = df["temp_c"].fillna(df["temp_c"].mean())
    #bckfill rain
    #df["will_it_rain"] = df["will_it_rain"].fillna(method="backfill")
    #backfill cloud
    #df["cloud"] = df["cloud"].fillna(method="backfill")
    df["will_it_rain"] = df["will_it_rain"].fillna(method="backfill")
    

    #set missing values to 0
    df["value"] = df["value"].fillna(0)
    df["is_day"] = df["timestamp"].apply(lambda x: 1 if " 06:" in x and " 22:" not in x else 0)
    df["precip_mm"] = df["precip_mm"].fillna(method="backfill")
    df["humidity"] = df["humidity"].fillna(df["humidity"].mean())

    
   

    



    #holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp_c"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    humidity = df["humidity"].values.astype(np.float32)
    humidity -= np.mean(humidity)  # normalize humidity by annual mean

    #rain = df["will_it_rain"].values.astype(np.float32)
    #snow = df["snow_1h"].values.astype(np.float32)
    #clouds = df["cloud"].values.astype(np.float32)
    precip_mm = df["precip_mm"].values.astype(np.float32)
    date_time = df["timestamp"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    

    hour_of_day = np.array([d.hour for d in date_time]).astype(np.float32)
    day_of_week = np.array([d.weekday() for d in date_time]).astype(np.float32)

    bikes_plus_120mins = df["bikes_plus_120mins"].values.astype(np.float32)
    #bikes_plus_60mins = df["bikes_plus_60mins"].values.astype(np.float32)

    df["bikes_plus_60mins"] = df['value'].shift(-1)
    bikes_plus_60mins = df["bikes_plus_60mins"].values.astype(np.float32)
    bikes_plus_60mins[-1] = bikes_plus_60mins[-2]

    df["bikes_plus_120mins"] = df['value'].shift(-2)
    bikes_plus_120mins = df["bikes_plus_120mins"].values.astype(np.float32)
    bikes_plus_120mins[-1] = bikes_plus_120mins[-2]
    bikes_plus_120mins[-2] = bikes_plus_120mins[-3]

    #print(df.isnull().sum())

    traffic_volume = df["value"].values.astype(np.float32)

    prev_hour_bikes = np.zeros_like(traffic_volume)
    prev_hour_bikes[1:] = traffic_volume[:-1]
    prev_hour_bikes[0] = traffic_volume[0]

    features = np.stack([temp,precip_mm,humidity, hour_of_day,day_of_week,prev_hour_bikes], axis=-1)

    # traffic_volume -= np.mean(traffic_volume)  # normalize
    # traffic_volume /= np.std(traffic_volume)  # normalize

    #return date_time,features, traffic_volume
    # if predict_hour == 1:
    #     return date_time,features, bikes_plus_60mins

    # elif predict_hour == 2:
    #     return date_time,features, bikes_plus_120mins

    df["traffic_in_1h"] = df['value'].shift(-1)
    y2 = df["traffic_in_1h"].values.astype(np.float32)
    y2[-1] = y2[-2]  # Fill last value with second to last value

    return date_time,features,traffic_volume,y2


def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)











# %%


# %%
from torch.utils.data import Dataset, DataLoader
import datetime as dt

class TrafficDataset(Dataset):
    def __init__(self, seq_len=8, exclude_timestamps=None,freqency='1h', predict_hour=1,inc=1,dataset_file="data/processed/"):
        date_time, x, y,y2 = load_bikes(predict_hour=predict_hour,dataset_file=dataset_file)
        date_time = np.array(date_time)

        x = np.array(x)
        y = np.array(y)
        y2 = np.array(y2)


        print("x.shape", x.shape)
        print("y.shape", y.shape)
        print("y2.shape", y2.shape)

        if exclude_timestamps is not None:
            exclude_timestamps = [dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in exclude_timestamps]
            #round to nearest hour
            exclude_timestamps = [ts.replace(minute=0, second=0) for ts in exclude_timestamps]
            print("Excluding {} timestamps".format(len(exclude_timestamps)))

            #prind exclude_timestamps
            indices_to_keep = [i for i, dt in enumerate(date_time) if dt not in exclude_timestamps]
            print("Keeping {} timestamps".format(len(indices_to_keep)))

            print("before", x.shape)
            
            date_time = date_time[indices_to_keep]
            x = x[indices_to_keep]
            print("after", x.shape)
            y = y[indices_to_keep]
            y2 = y2[indices_to_keep]

        self.x, self.y = cut_in_sequences(x, y, seq_len, inc=inc)
        self.x, self.y2 = cut_in_sequences(x, y2, seq_len, inc=inc)
        print("dataset x.shape", self.x.shape)
        print("date_time.shape", date_time.shape)

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        
        #return self.x[:, idx], self.y[:, idx].reshape(-1, 1)
        #return self.x[:, idx], self.y[:, idx].reshape(-1, 1), self.y2[:, idx].reshape(-1, 1)
        y_combined = np.stack([self.y[:, idx].reshape(-1, 1), self.y2[:, idx].reshape(-1, 1)], axis=-1)
        return self.x[:, idx], y_combined




# %% [markdown]
# For the training we will use Pytorch-Lightning, thus we have to define our learner module. 

# %%
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        
        # calculate MAE for each target separately
        mae1 = nn.L1Loss()(y_hat[..., 0], y[..., 0])
        mae2 = nn.L1Loss()(y_hat[..., 1], y[..., 1])
        self.log("tm1", mae1, prog_bar=True)
        self.log("tm2", mae2, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        
        # calculate MAE for each target separately
        mae1 = nn.L1Loss()(y_hat[..., 0], y[..., 0])
        mae2 = nn.L1Loss()(y_hat[..., 1], y[..., 1])
        self.log("vm1", mae1, prog_bar=True)
        self.log("vm2", mae2, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    



# Next we define some toy dataset and create the corresponding DataLoaders

if __name__ == "__main__":


    dataset_folder='/d/hpc/home/ld8435/liquid_time_constant_networks/experiments_with_ltcs/data/stations_data_preprocessed/'

    dataset_file = "/d/hpc/home/ld8435/liquid_time_constant_networks/experiments_with_ltcs/data/bicikelj/bicikelj_one.csv"

    

    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torch.utils.data import DataLoader








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
    
    import os
    

    csv_files = [fileee for fileee in os.listdir(dataset_folder) if fileee.endswith(".csv")]

    print(len(csv_files))



    prediction_df = pd.DataFrame(columns=["timestamp"])
    prediction_df["timestamp"] = timestamps
    prediction_df["timestamp"] = pd.to_datetime(prediction_df["timestamp"])

   

    for csv_file in csv_files:
        dataset_file = os.path.join(dataset_folder, csv_file)

        # Instantiate the TrafficDataset
        dataset = TrafficDataset(seq_len=8, exclude_timestamps=timestamps,freqency="1h",predict_hour=1,inc=2,dataset_file=dataset_file)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        # Get a batch of data
        data_x, data_y = next(iter(dataloader))

        # Calculate the sizes of the train, validation, and test sets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset into train, validation, and test sets
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders for train, validation, and test sets
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)


        # Example usage of the train, validation, and test DataLoaders
        # train_batch_x, train_batch_y = next(iter(train_dataloader))
        # val_batch_x, val_batch_y = next(iter(val_dataloader))
        # test_batch_x, test_batch_y = next(iter(test_dataloader))

        # Convert to torch.Tensor
        data_x = torch.Tensor(data_x)
        data_y = torch.Tensor(data_y)

        # Let's visualize the training data
        # sns.set()
        # plt.figure(figsize=(6, 4))

        # The visualization might change depending on the structure of your data
        # Here, I'm assuming your data has two features and one target variable.
        # Modify accordingly if your data structure is different.
        # plt.plot(data_x[0, :, 4], label="feaure x")
        # plt.plot(data_y[0, :,0], label="Target output")

        # plt.title("Training data")
        # plt.legend(loc="upper right")
        # plt.show()

        # #plotly plot

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

        wiring = AutoNCP(64, 2)  # 16 units, 1 motor neuron

        ltc_model = LTC(6, wiring, batch_first=True)
        learn = SequenceLearner(ltc_model, lr=0.01)

        # Define early stopping criteria
        early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,  # minimum value of the change in val_loss for the change to be considered an improvement
        patience=20,  # number of epochs with no improvement after which training will be stopped
        verbose=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            logger=pl.loggers.CSVLogger("log"),
            max_epochs=600,
            gradient_clip_val=1,  # Clip gradient to stabilize training
            devices=1,  # Use GPU
            accelerator='gpu',
            log_every_n_steps=1,
            callbacks=[early_stop_callback, lr_monitor]
        )

        trainer.fit(learn, train_dataloader, val_dataloader)
        results = trainer.test(learn, test_dataloader)
        print(results)


        #get the nearest timestamp in the dataset
        date_time, feature, target1,target2 = load_bikes(predict_hour=1,dataset_file=dataset_file)
        #find the nearest timestamp in date_time that is not the same as the timestamp in the list
        #find the index of the timestamp in the list
        seq_len = 24
        nearest_timestamps = []
        for timestamp in timestamps:
            nearest_timestamp = min(date_time, key=lambda x: abs(x - pd.to_datetime(timestamp)))
            nearest_timestamps.append(nearest_timestamp)
            #print(f"nearest timestamp to {timestamp} is {nearest_timestamp}")

        feats=[]

        #get indexes of the nearest timestamps
        indexes_of_nearest_timestamps = []
        for timestamp in nearest_timestamps:
            indexes_of_nearest_timestamps.append(date_time.index(timestamp))

        #print(indexes_of_nearest_timestamps)

        x_sequences = []
        for index in indexes_of_nearest_timestamps:
            x_sequences.append(feature[max(index-seq_len+1, 0):index+1])


        x_sequences = np.array(x_sequences)

        # Ensure your model is in evaluation mode
        ltc_model.eval()

        # Convert x_sequences to torch tensor
        x_sequences_tensor = torch.from_numpy(x_sequences).float()

        # If your model is on GPU, move the tensor to the GPU
        # x_sequences_tensor = x_sequences_tensor.to(device)

        # Perform prediction
        with torch.no_grad():
            predictions_tuple = ltc_model(x_sequences_tensor)

        predictions = predictions_tuple[0]
        # Convert predictions back to numpy
        predictions = predictions.cpu().numpy()


        final=[]
        for i in range(0, len(predictions)):
            if i % 2 == 0:
                final.append(abs(int(predictions[i, seq_len-1, 0])))
            else:
                p2=abs(int(predictions[i, seq_len-1, 1]))
                p1=abs(int(predictions[i, seq_len-1, 0]))
                final.append(p2)




        prediction_df[csv_file.replace("_preprocessed.csv", "").replace("_", "/")] = final
        #save the predictions to csv
        prediction_df.to_csv("/d/hpc/home/ld8435/liquid_time_constant_networks/predictionsss.csv", index=False)



        # df = pd.DataFrame(final, columns=["prediction"])
        # df["timestamp"] = timestamps
        # df.to_csv("/d/hpc/home/ld8435/liquid_time_constant_networks/predictions.csv", index=False)
        # #open bicikelj_terfrfst_xgb_1h_fullds_new_pred_method_new_feats3
        # to_Send=pd.read_csv("/d/hpc/home/ld8435/liquid_time_constant_networks/pppp.csv")
        # to_Send["timestamp"]=pd.to_datetime(to_Send["timestamp"])

        # #set colum PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE valuses to predictions rounded to int and  negative beeeing 0

        # to_Send["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"]=final
        # to_Send["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"]=to_Send["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"].clip(lower=0)

        # #save to csv
        # to_Send.to_csv("/d/hpc/home/ld8435/liquid_time_constant_networks/ok.csv", index=False)







    # Plot the first sequence predictions against actual values
    # plt.figure(figsize=(6, 4))
    # plt.plot(predictions[:, seq_len-1, 0], label="Model output 1h")
    # plt.plot(predictions[:, seq_len-1, 1], label="Model output 2h")

    # plt.title("Model Predictions")
    # plt.legend(loc="upper right")
    # plt.show()

















# %%
#make predictions on timestamps


# %%
# #plotyl plot of predictions

# # Plot the predictions using plotly
# import plotly.graph_objects as go

# from sklearn.metrics import mean_absolute_error


# for i in range(0, len(predictions)):
#     fig = go.Figure()
#     if i % 2 == 0:
        
#         print(f"prediction for {timestamps[i]} is {predictions[i, seq_len-1, 0]}")
       
#         fig.add_trace(go.Scatter(y=predictions[i, :, 0],
#                             mode='lines',
#                             name='predictions'))
#     else:
#         print(f"prediction for {timestamps[i]} is {predictions[i, seq_len-1, 1]}")
#         fig.add_trace(go.Scatter(y=predictions[i, :, 1],
#                             mode='lines',
#                             name='predictions'))

#     #add features x_sequences_tensor 0
#     fig.add_trace(go.Scatter(y=x_sequences_tensor[i, :, -1],
#                         mode='lines',
#                         name='features'))
    





#     fig.show()

# %%


# %%
#save tocsv as timestamp, prediction
#convert to dataframe
