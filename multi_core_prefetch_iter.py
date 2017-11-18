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
class PrefetchIter(object):
    '''
    classdocs
    '''


    def __init__(self, data_iter,num_processes = 2):
        '''
        Constructor
        '''
        self.data_iter = data_iter
        self.batch_size = dict(self.data_iter.provide_data)['data'][0]
        self.index = 0

        self.index_lock = multiprocessing.Lock()
        self.queue = multiprocessing.Queue(num_processes * 2)        
        self.num_processes = num_processes      
        self.processes_list = []  
        self.reset()
    def worker(self,data_iter,lock,q):  
        logging.info("starting a process to continue.")
        try:
            while self.need_to_continue: 
                with self.index_lock:
                    index = self.index
                try:
                    da = data_iter[index]
                    while self.need_to_continue:
                        try:
                            self.queue.put(da,block = False)
                            break
                        except Queue.Full:
                            time.sleep(0.1)
#                             logging.debug("FULL")
                            pass
                        with self.index_lock:
                            self.index += self.batch_size
                except IndexError:
                    self.reach_end = True
                    return
        except Exception as e:
            logging.error(e)
    def __next__(self):
        if self.reach_end:
            for p in self.self.processes_list:
                p.join()  
        if self.queue.empty() and self.reach_end:
            raise StopIteration
        else:            
            da = self.queue.get()
            da.data = [mx.nd.array(d) for d in da.data]
            da.label = [mx.nd.array(d) for d in da.label]
            return da
    def reset(self):
        self.need_to_continue = False
        try:
            '''
            Ensure all preocesses has ended.
            '''            
            for p in self.self.processes_list:
                p.join()
        except AttributeError:
            pass

        while not self.queue.empty():
            try:
                self.queue.get(block  =False)
            except Queue.Empty:
                break
        self.processes_list = []
        self.need_to_continue = True
        self.reach_end = False
    
        for _ in range(self.num_processes):
            p = multiprocessing.Process(target = self.worker,args = (copy.copy( self.data_iter),0,0))
            p.start()
            self.processes_list.append(p)
        with self.index_lock:
            self.index = 0
        
    def next(self):
        return self.__next__()
    def __iter__(self):
        return self
    
