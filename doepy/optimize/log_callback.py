"""
MIT License

Copyright (c) 2019 Simon Olofsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import pickle
from pathlib import Path

class LogCallback:
	def __init__ (self, save_folder=None, savefile_prefix='slsqp_log',
	              save_gradients=False):
		# Save gradients?
		self.save_gradients = save_gradients

		# Time stamp
		t = time.localtime()
		t = "%d_%d_%d_%d%d" %(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)

		if save_folder is None:
			self.logfile = Path()
		else:
			self.logfile = Path(save_folder)
			assert self.logfile.is_dir(), 'Invalid directory: %s'%self.logfile

		self.logfile /= '%s_%s.pickle'%(savefile_prefix, t)
		
	def load (self):
		with open(self.logfile, 'rb') as fil:
			data = pickle.load(fil)
		return data
	
	def save (self, data):
		with open(self.logfile, 'wb') as fil:
			pickle.dump(data, fil, pickle.HIGHEST_PROTOCOL)

	def __call__ (self, u, f, c, df, dc):
		# LOAD
		if self.logfile.is_file():
			data = self.load()
		else:
			data = {'u':[], 'f':[], 'c':[], 'n':0}
			if self.save_gradients:
				data.update({'df':[], 'dc':[]})

		# UPDATE
		data['u'].append(u)
		data['f'].append(f)
		data['c'].append(c)
		data['n'] += 1
		if self.save_gradients:
			data['df'].append(df)
			data['dc'].append(dc)

		# SAVE
		self.save(data)