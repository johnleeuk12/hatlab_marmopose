#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:07:38 2026

@author: jlee629
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:23:58 2026

@author: jlee629
"""

import matplotlib.pyplot as plt
import numpy as np
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_ # Or np.bool
import os


# %%
data_dir = '/media/jlee629/D/Neural Data/Phy'
session_name = 'TS_05_11'

fpath1 = os.path.join(data_dir,session_name)

n_channels = 99      # total channels in the .bin, including analog
n_samples  = 30000*60

data = np.fromfile(os.path.join(fpath1,'rawdat.bin'), dtype=np.int16,
                   count=n_samples * n_channels).reshape(n_samples, n_channels).T
# data.shape == (n_channels, n_samples)

# sync = data[97]  
fig, axes = plt.subplots(20, 1, figsize=(18, 5))

for ch in np.arange(79,99):  
    axes[ch-79].plot(data[ch])
# axes.plot(data[97])
# axes.plot(data[98])


# %%

from neo.rawio import BlackrockRawIO

reader = BlackrockRawIO(filename=os.path.join(fpath1,'datafile001.nev'))   # no extension — it pairs .nev/.nsX by basename
reader.parse_header()
print(reader)     

# what's actually in the file
print(reader.header['spike_channels'])         # unit IDs, names
print(reader.header['event_channels'])         # digital_input, comments, etc.
print(reader.raw_annotations['blocks'][0])     # nev basic header: date, timestamp resolution

# digital events (sync pulses, TTLs, Cerebus serial/digital port)
ts, durations, labels = reader.get_event_timestamps(block_index=0, seg_index=0,
                                                    event_channel_index=0)



# %%

from neo.rawio import BlackrockRawIO

reader = BlackrockRawIO(filename=os.path.join(fpath1,'datafile001.ns6'), nsx_to_load=6)
# reader.parse_header()

# for ch in reader.header['signal_channels']:
#     print(ch['id'], ch['name'], ch['units'], ch['gain'], ch['offset'])
    
reader.parse_header()

chans = reader.header['signal_channels']
idx = int(np.where(chans['name'] == 'ainp1')[0][0])   # positional index in the stream
print(chans[idx])

x = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                                  i_start=1_100_000, i_stop=1_500_000,
                                  stream_index=0, channel_indexes=[idx]).ravel()
print(x.min(), x.max())
d = np.abs(np.diff(x.astype(np.int32)))
print(np.where(d > d.std() * 10)[0][:10])   # candidate edges, in samples from i_start




fig, axes = plt.subplots(1, 1, figsize=(18, 5))

axes.plot(x)

# %%


from datetime import datetime, timezone

from zoneinfo import ZoneInfo
reader = BlackrockRawIO(filename=os.path.join('/media/jlee629/D/Neural Data/Raws/TS_raw1_05_11/TS_raw1_05_11','datafile001.ns6'), nsx_to_load=6)

reader.parse_header()

blk = reader.raw_annotations['blocks'][0]
t0 = blk['rec_datetime']          # datetime, millisecond resolution
print(t0)

t0_epoch = t0.timestamp()   
trig = 1778507292.9492
print(datetime.fromtimestamp(trig,tz = timezone.utc))                    # local time on this machine

