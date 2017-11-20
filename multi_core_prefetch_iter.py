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
from pydoc import serve
import multiprocessing.managers as mpm
import threading

class PrefetchIter(object):
    '''
    classdocs
    '''

    def __init__(self, data_iter,num_processes = 16):
        '''
        Constructor
        '''
        self.data_iter = data_iter
        self.batch_size = dict(self.data_iter.provide_data)['data'][0]
        
        from multiprocessing import Manager
        m= Manager()
        self.queue = m.Queue(maxsize = 4 * num_processes)
        self.index = multiprocessing.Value('i',0,lock  = True)

        self.index_lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.reach_end = multiprocessing.Value('i',0)
        self.processes_count = multiprocessing.Value('i',0)
        self.num_processes = num_processes      
        self.processes_list = []  


        self.reset()
        
    def worker(self,data_iter,lock,index,stop_event,reach_end,batch_size,processes_count,q):  
#         class SharedQueueManager(mpm.BaseManager):pass
#         q = Queue.Queue
#         SharedQueueManager.register('Queue',lambda:q)
#         mgr = SharedQueueManager(address = ('localhost',12345))
#         mgr.connect()
#         q = mgr.Queue()
        
        with processes_count.get_lock():
            processes_count.value += 1
            logging.debug("starting a process to continue.{0}".format(processes_count.value))
        try:
            while not stop_event.is_set(): 
                try:
                    with index.get_lock():
                        ind = int(index.value)
                        index.value += batch_size
                    logging.debug("fetching...{0}".format(ind))                        
                    da = data_iter[ind]
                    while not stop_event.is_set():
                        #logging.debug("FULL_{0}".format(stop_event.is_set()))
                        try:
                            q.put(da,block = False)
                            break
                        except Queue.Full:
                            time.sleep(0.1)
#                            logging.debug("FULL_{0}".format(stop_event.is_set()))
                            pass
                except IndexError as e:
                    logging.exception(e)
                    reach_end.value = 1
                    logging.info("exist process...")
                    break
        except Exception as e:
            logging.exception(e)
        with processes_count.get_lock():
            processes_count.value -= 1
            logging.info("Existing a process.{0}".format(processes_count.value))

    def __next__(self):
        if self.reach_end.value:
            self.stop_event.set()
            while self.processes_count.value !=0:
                time.sleep(0.1)
            '''
            join each process, to avoid dead process.
            '''
            for p in self.processes_list:
                p.join()

            if self.queue.empty():
                raise StopIteration
            else:
                da = self.queue.get()
                da.data = [mx.nd.array(d) for d in da.data]
                da.label = [mx.nd.array(d) for d in da.label]
                return da                
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
            while self.processes_count.value !=0:
                time.sleep(0.1)
            '''
            join each process, to avoid dead process.
            '''
            for p in self.processes_list:
                p.join()
        except AttributeError:
            pass

        while not self.queue.empty():
            try:
                self.queue.get(timeout = 0.1)
                time.sleep(0.1)
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
                self.index,
                self.stop_event,
                self.reach_end,
                self.batch_size,
                self.processes_count,
                self.queue
                ))
            p.daemon = True
            p.start()
            self.processes_list.append(p)
        
    def next(self):
        return self.__next__()
    def __iter__(self):
        return self
    
