# -*- coding:utf-8 -*-

#  Set and config the log modules


import logging
import os
import time

class Logger:
    def __init__(self,loggername):

    	logs_dir = "../logs"
    	if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
    		pass
    	else:
    		os.mkdir(logs_dir)



    	timestamp = time.strftime("%Y-%m-%d", time.localtime())
    	logname = '%s.log' % timestamp
    	log_file_path = os.path.join(logs_dir, logname)

    	self.logger = logging.getLogger(log_file_path)
    	self.logger.setLevel(logging.DEBUG)


    	fh = logging.FileHandler(log_file_path,encoding = 'utf-8')
    	fh.setLevel(logging.DEBUG)


    	ch = logging.StreamHandler()
    	ch.setLevel(logging.DEBUG)
    	
    	# set the format of logs

    	formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')



    	fh.setFormatter(formatter)
    	ch.setFormatter(formatter)


    	self.logger.addHandler(fh)
    	self.logger.addHandler(ch)


    def get_log(self):
    	return self.logger  
