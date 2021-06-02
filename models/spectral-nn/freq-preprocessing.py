import pandas as pd
import numpy as np
import json
from scipy.fft import rfft, rfftfreq, dct
from glob import glob
import os
import sys
from time import time
import multiprocessing as mp
from multiprocessing import Manager, Pool, Lock
from math import sqrt
from pprint import pprint
from bins import Bins
import matplotlib.pyplot as plt
from numbers import Real

# set multiprocessing to use 'fork' instead of 'spawn'
# issue details here: https://github.com/ipython/ipython/issues/12396
# note, map_async will still hang
# mp.set_start_method('fork')

# relative path to the raw folder of the wisdm dataset
RAW_DIR_PATH = "../wisdm-dataset/raw/"


# some global information
col_names = ['id', 'label', 'timestamp', 'x', 'y', 'z']
dtypes = ['int64', 'object', 'int64', 'float64', 'float64', 'float64']

# generator for calculating the average iteratively
def _avg():
    avg, val, n = 0, 0, 1
    while True:
        val = (yield avg)
        if not isinstance(val, Real):
            avg = 0
            val = 0
            n = 1
            continue
        # recurrence relation
        avg = (avg * (n - 1) / n) + val / n
        n += 1

# generator for calculating the variance iteratively
def _var():
    avg, var, va, n = 0, 0, 0, 1
    gen_avg = _avg()
    next(gen_avg)
    while True:
        val = (yield var)
        if not isinstance(val, Real):
            avg = 0
            var = 0
            n = 1
            gen_avg.send('reset')
            continue
        new_avg = gen_avg.send(val)
        # recurrence relation
        var = var + avg**2 - new_avg**2 + (val**2 - var - avg**2) / n + 1
        avg = new_avg
        n += 1

# function that reads and processes a file and adds it to
#   the results dictionary with the subject id as key
def process_file(filepath):
    # user feedback
    filename = os.path.split(filepath)[1]
    # print(f"processing file: {filename}", flush=True)
    tf = time()

    # get the subject id from the filename
    sid = filename.split('_')[1]
    
    with open(filepath, 'r') as file:

        # process raw text files into a json-readable dictionary format
        raw = []
        for line in file.readlines():
            # clean the line and convert to a list
            data = line.rstrip(';\n').split(',')

            # turn the list into a dictionary and append it to raw list
            # raw.append(dict(zip(col_names, data)))
            
            raw.append(data)

    # convert raw to a json string and write it to a dataframe
    # raw_df = pd.read_json(json.dumps(raw),
    #                       orient='records',
    #                       dtype=dict(zip(col_names, dtypes)),
    #                       convert_dates=False
    #                      )

    raw_df = pd.DataFrame(raw, columns=col_names).astype(dict(zip(col_names, dtypes)))

    # generate groupby object by the label
    grouped_df = raw_df.groupby(raw_df.label)

    # construct dictionary of each group with unique label
    raw_dict = {}
    for l in grouped_df.groups:

        # get the dataframe for the label
        df = grouped_df.get_group(l).copy(deep=True)

        # reset the index and drop the 'index' label
        df = df.reset_index().drop(labels=['index'], axis=1)

        # initialize generators
        avg_gen = _avg()
        var_gen = _var()
        
        # compute average and standard deviation of sample spacing
        average = next(avg_gen)
        variance = next(var_gen)
        for i in range(1, len(df['timestamp'])):
            diff = int(df['timestamp'][i] - df['timestamp'][i-1])
            average = avg_gen.send(diff)
            variance = var_gen.send(diff)
        
        # add to dictionary
        raw_dict[l] = {
            'df': df,
            'avg ts': average,
            'stdv ts': sqrt(variance)
        }
    
    print(f"{filename.split('_')[1]} processed in {time()-tf}\n", flush=True, end='')
    
    return (sid, raw_dict)


def process_freq(df, ts, label):
    tp = time()
    # this function assumes that the data is mostly evenly spaced. since in reality the
    # data we are working with is not, there will probably be some error in the final results
    # how badly this propagates is yet to be seen
    
    # average sampling rate
    fs = 1/ts 
    
    # code is largely taken from this excellent blog post:
    # https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    
    # pre-emphasis (may be excluded later)
    alpha = 0.97
    x_emph = np.append(x[0], x[1:] - alpha * x[:-1])
    y_emph = np.append(y[0], y[1:] - alpha * y[:-1])
    z_emph = np.append(z[0], z[1:] - alpha * z[:-1])
    
    # framing
    # the principle is the same, but since human motion is much lower frequency than 
    # speech, we will pick much larger frame size and stride than the article. 
    # our whole data is 3 min, so to save on computation time, and since we will be 
    # averaging the frames together at the end for simplicity, we'll choose (in seconds)
    fsize = 60
    fstride = 30
    
    flen = int(round(fsize * fs))
    fstep = int(round(fstride * fs))
    
    siglen = len(x_emph)  # they should all be the same
    fnum = int(np.ceil(float(np.abs(siglen - flen)) / fstep))
    
    # zero pad the end of the signal so we don't have indexing error
    # and so we don't have to truncate
    paddedlen = fnum * fstep + flen
    pad = np.zeros((paddedlen - siglen))
    x_pad = np.append(x_emph, pad)
    y_pad = np.append(y_emph, pad)
    z_pad = np.append(z_emph, pad)
    
    idxs = np.tile(np.arange(0, flen), (fnum, 1)) + np.tile(np.arange(0, fnum * fstep, fstep), (flen, 1)).T
    idxs = idxs.astype(np.int32, copy=False)
    x_frames = x_pad[idxs]
    y_frames = y_pad[idxs]
    z_frames = z_pad[idxs]
    
    # apply window function (hamming window)
    # reasons are as stated in the blog post
    x_frames *= np.hamming(flen)
    y_frames *= np.hamming(flen)
    z_frames *= np.hamming(flen)
    
    # compute N-point FFT and compute power spectrum
    # thing says use 256 or 512, but i'm going to use the full frame length
    NFFT = flen
    x_mag = np.absolute(rfft(x_frames, n=NFFT))    # magnitude of fft
    y_mag = np.absolute(rfft(y_frames, n=NFFT))
    z_mag = np.absolute(rfft(z_frames, n=NFFT))
    
    x_pow = (((x_mag) ** 2) / NFFT)
    y_pow = (((y_mag) ** 2) / NFFT)
    z_pow = (((z_mag) ** 2) / NFFT)
    
    # apply filter banks to get the bands of the power spectrum
    # unlike the blog, we are not converting to the mel spectrum
    # so although this process is very similar to MFCC, it isn't the same.
    nfilt = 16    # interested in 0 to 10 Hz
    # note we go to 11 Hz because the last triangle corresponding to 10 ends on it.
    bins = np.floor((NFFT + 1) * np.linspace(0, 9, nfilt+2) / fs)
    
    # note that the filter bank ignores 0 frequency since it is just DC term
    # the original post uses overlapping triangle filters, but these introduce 
    # significant correlation. to decorrelate, he uses an additional DCT, though
    # this again introduces more computation time. (hopefully not too much...)
    
    bins_edges = np.array([np.floor((bins[i] + bins[i+1]/2)) for i in range(len(bins)-1)])
    
    # constructing the filter
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    
    # rectangle filter bank
    for i in range(1, nfilt+1):
        f_left = int(bins[i])
        f_right = int(bins[i+1])

        for k in range(f_left, f_right):
            fbank[i-1, k] = 1 / (f_right - f_left)
    
    # triangle filter bank
    # for i in range(1, nfilt + 1):
    #     f_left = int(bins[i - 1])
    #     f = int(bins[i])
    #     f_right = int(bins[i+1])

    #     for k in range(f_left, f):
    #         fbank[i - 1, k] = (k - bins[i-1]) / (bins[i] - bins[i-1])
    #     for k in range(f, f_right):
    #         fbank[i - 1, k] = (bins[i+1] - k) / (bins[i + 1] - bins[i])
    
    # applying the filter
    x_pow_fil = np.dot(x_pow, fbank.T)
    y_pow_fil = np.dot(y_pow, fbank.T)
    z_pow_fil = np.dot(z_pow, fbank.T)
    
    x_mag_fil = np.dot(x_mag, fbank.T)
    y_mag_fil = np.dot(y_mag, fbank.T)
    z_mag_fil = np.dot(z_mag, fbank.T)
    
    # get rid of zeros for numerical stability
    x_pow_fil = np.where(x_pow_fil == 0, np.finfo(float).eps, x_pow_fil)
    y_pow_fil = np.where(y_pow_fil == 0, np.finfo(float).eps, y_pow_fil)
    z_pow_fil = np.where(z_pow_fil == 0, np.finfo(float).eps, z_pow_fil)
    
    # conversion to decibels
    x_db = 20 * np.log10(x_pow_fil)
    y_db = 20 * np.log10(y_pow_fil)
    z_db = 20 * np.log10(z_pow_fil)
    
    # compute cepstral coefficients (keep 12 as in speech recognition, since 
    # the first coefficient represents the DC bias of the frequencies, and 
    # there shouldn't be too much variation in frequencies, but this might not be the case)
    x_cc = dct(x_db, axis=1, norm='ortho')[:, 1:16]
    y_cc = dct(y_db, axis=1, norm='ortho')[:, 1:16]
    z_cc = dct(z_db, axis=1, norm='ortho')[:, 1:16]
    
    # compute the averages of returned entities
    x_mag_avg = np.mean(x_mag_fil, axis=0)
    y_mag_avg = np.mean(y_mag_fil, axis=0)
    z_mag_avg = np.mean(z_mag_fil, axis=0)
    x_pow_avg = np.mean(x_pow_fil, axis=0)
    y_pow_avg = np.mean(y_pow_fil, axis=0)
    z_pow_avg = np.mean(z_pow_fil, axis=0)
    x_cc_avg = np.mean(x_cc, axis=0)
    y_cc_avg = np.mean(y_cc, axis=0)
    z_cc_avg = np.mean(z_cc, axis=0)
    
    # will not apply liftering because we might not need it.
    # return the averaged results (hopefully they look good)
    results = {
        'xmag': x_mag_avg,
        'ymag': y_mag_avg,
        'zmag': z_mag_avg,
        'xpow': x_pow_avg,
        'ypow': y_pow_avg,
        'zpow': z_pow_avg,
        'xcc': x_cc_avg,
        'ycc': y_cc_avg,
        'zcc': z_cc_avg
    }
    
    # print(f"Elapsed Time: {time() - tp}")
    return (label, results)


# function that writes processed data to file
# while dropping bad data
# define bad data as:
#   std dev sampling period > 10% average sampling period
#   average sampling period > 55 ms
feat_order = ['xmag', 'ymag', 'zmag', 'xpow', 'ypow', 'zpow', 'xcc', 'ycc', 'zcc']
def write_processed(args):
    (data, filename) = args
    sid = os.path.split(filename)[1].split('_')[0]
    t = time()
    # generate argument list
    args = [(data[l]['df'], data[l]['avg ts'] * (10**-9), l) for l in data.keys() \
            if 'avg ts' in data[l] \
            and data[l]['avg ts'] <= 55000000 \
            and data[l]['stdv ts'] <= data[l]['avg ts'] * 0.1]

    # this is a pretty fast process, so we can just run it for all of them in a file
    results = [process_freq(*arg) for arg in args]

    results = dict(results)
    
    with open(filename, 'w') as file:
        s = ','.join([
            'sid',
            ','.join([f'xmag{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'ymag{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'zmag{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'xpow{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'ypow{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'zpow{round(i/2, 1)}' for i in range(1,17)]),
            ','.join([f'xcc{i}' for i in range(1,16)]),
            ','.join([f'ycc{i}' for i in range(1,16)]),
            ','.join([f'zcc{i}' for i in range(1,16)]),
            'label'
        ]) + '\n'
        file.write(s)
        for label, rdict in results.items():
            s = f"{sid},"
            for feat in feat_order:
                s += ",".join([f"{val}" for val in rdict[feat]])
                s += ','
            s += label + '\n'
            file.write(s)
    
    print(f"{os.path.split(filename)[1]} written to file in {time()-t} seconds!", flush=True)


def main():
    # loop through each subdirectory
    subdirs = ["phone/accel/", "phone/gyro/", "watch/accel/", "watch/gyro/"]
    all_results = {}
    for subdir in subdirs:
        
        # graph a list of all files in the subdirectory
        filenames = glob(f"{RAW_DIR_PATH}{subdir}/*")
        
        # remove any empty strings
        filenames = [name for name in filenames if name]
        
        # sort the list
        filenames.sort()
        
        # process data with multiprocessing
        
        # was going to use a manager, but a pool might be faster anyways
        
        # timing information
        print(f"\nHandling Subdirectory: {subdir}\n")
        t = time()
        
        with Pool(processes=2) as pool:
            
            # map processing function and store results in list
            results = pool.map(process_file, filenames)
        
        # data = []
        # for filepath in filenames:
        #     print(f"Processing File: {os.path.split(filepath)[1]}", flush=True)
        #     tf = time()
        #     data.append(process_file(filepath))
        #     print(f"Done! Elapsed Time: {time()-t}")
        
        all_results[subdir] = dict(results)
        
        # timing
        elapsed = time() - t
        print(f"\nDone! Elapsed Time: {elapsed}")

    # make a folder for processed data
    if not os.path.exists("_spectral"):
        os.makedirs('_spectral')
        
    for subdir in subdirs:
        
        context = subdir.strip('/').split('/')
        base = "_".join(context) + ".txt"
        
        args = [(data, "_spectral/" + str(sid) + '_' + base) for sid, data in all_results[subdir].items()]
        
        t = time()
        
        print("Starting Processes...", flush=True)
        
        # again apply multiprocessing to handle a lot of data
        with Pool(processes=2) as pool:
            
            # use starmap for multiple arguments
            pool.map(write_processed, args)
            
        print(f"\nDone! Finished in {time()-t} seconds")

    print("\n\nFinished Spectral Processing! Run freq-model.py for simple ANN (poor accuracy)")

if __name__ == "__main__":
    main()