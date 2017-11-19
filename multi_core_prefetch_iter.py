'''
Created on Nov 18, 2017

@author: kohill
'''
import multiprocessing
import copy
from numpy.linalg.linalg import multi_dot
import logging
import mxnet as mx
import time
import Queue
import numpy as np
class PrefetchIter(object):
    '''
    classdocs
    '''


    def __init__(self, data_iter,num_processes = 8):
        '''
        Constructor
        '''
        self.data_iter = data_iter
        self.batch_size = dict(self.data_iter.provide_data)['data'][0]
        
        from multiprocessing import Manager
        
        self.index = multiprocessing.Value('i',0,lock  = True)

        self.index_lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.reach_end = multiprocessing.Value('i',0)
        self.queue = multiprocessing.Queue(num_processes * 2)        
        self.num_processes = num_processes      
        self.processes_list = []  
        self.reset()
        
    def worker(self,data_iter,lock,q,index,stop_event,reach_end,batch_size):  
        logging.info("starting a process to continue.")
        try:
            while not stop_event.is_set(): 
                try:
                    with index.get_lock():
                        ind = int(index.value)
                        index.value += batch_size
                    logging.debug("fetching...{0}".format(ind))                        
                    da = data_iter[ind]
                    while not stop_event.is_set():
                        logging.debug("FULL_{0}".format(stop_event.is_set()))
                        try:
                            q.put(da,block = False)
                            break
                        except Queue.Full:
                            time.sleep(0.1)
                            logging.debug("FULL_{0}".format(stop_event.is_set()))
                            pass
                except IndexError as e:
                    logging.exception(e)
                    reach_end.value = 1
                    logging.info("exist process...")
                    return
        except Exception as e:
            logging.exception(e)
    def __next__(self):
        need_to_wait = False
        if self.reach_end.value:
            self.stop_event.set()
            for p in self.processes_list:
                if p.is_alive():
                    need_to_wait = True
        if self.queue.empty() and not need_to_wait:
            raise StopIteration
        else:            
            da = self.queue.get()
            da.data = [mx.nd.array(d) for d in da.data]
            da.label = [mx.nd.array(d) for d in da.label]
            return da
    def reset(self):
        self.stop_event.set()
        try:
            '''
            Ensure all preocesses has ended, but before that, we need to ensure that queue has no data.
            '''            
            all_p_stoped = False
            while not all_p_stoped:
                all_p_stoped = True
                for p in self.processes_list:   
                    if p.is_alive():
                        all_p_stoped = False
                time.sleep(1)
        except AttributeError:
            pass

        while not self.queue.empty():
            try:
                self.queue.get(block  =False)
            except Queue.Empty:
                break
        self.processes_list = []
        self.stop_event.clear()

        self.reach_end.value = 0
        with self.index_lock:
            self.index.value = 0

        for _ in range(self.num_processes):
            p = multiprocessing.Process(target = self.worker,args = (
                copy.copy( self.data_iter),
                self.index_lock,
                self.queue,
                self.index,
                self.stop_event,
                self.reach_end,
                self.batch_size
                ))
            p.daemon = True
            p.start()
            self.processes_list.append(p)
        
    def next(self):
        return self.__next__()
    def __iter__(self):
        return self
    
