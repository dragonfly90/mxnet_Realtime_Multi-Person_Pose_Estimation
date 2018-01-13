

train_deeplab:
	export PYTHONPATH=$(PYTHONPATH):../cython:.. &&	echo $(PYTHONPATH) && \
		cd deeplab && python2 train_deeplab.py

