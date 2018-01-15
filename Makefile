
all:
	echo "do nothing..."
train_deeplab:
	export PYTHONPATH=$(PYTHONPATH):../cython:.. &&	echo $(PYTHONPATH) && \
		cd deeplab && python2 train_deeplab.py
eval:
	export PYTHONPATH=$(PYTHONPATH):../cython:.. &&	echo $(PYTHONPATH) && \
		cd deeplab && python2 evaluation_coco.py

clean:
	rm -f deeplab/logs/*
	rm -f deeplab/models/*