from nnmodel import Reader, Model
import sys
import pickle

# Example using phone_accel data

# Apply the Reader class:
df = Reader("models/phone_accel/",mode='d').df # read entire directory into dataframe

# Apply the Model class:
# Evaluate model with 70/30 train/test split

# list attributes and label of choice
attributes = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1',
       'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG',
       'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
       'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'RESULTANT']
label = 'ACTIVITY'
testsize = 0.3 # use a 70/30 train/test split

# initiate and run model
model = Model(df)
model.preprocess(attributes, label, testsize)
model.train(kfold=False) # Use train/test split for training instead of kfold

# Save model for later prediction
estimator = model.estimator
estimator.model.save('models/nn.h5')
pickle.dump(estimator.classes_, open( 'models/nn_classes.p', 'wb' ))
pickle.dump(model.encoder, open( 'models/nn_encoder.p', 'wb' ))

sys.stdout.flush()
sys.stderr.flush()