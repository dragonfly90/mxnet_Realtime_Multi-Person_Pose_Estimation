
all:
	echo "do nothing..."
train_deconv:
	export PYTHONPATH=$(PYTHONPATH):../cython:.. &&	echo $(PYTHONPATH) && \
		cd deconv && python2 train_deconv.py
eval:
	export PYTHONPATH=$(PYTHONPATH):../cython:.. &&	echo $(PYTHONPATH) && \
		cd deeplab && python2 evaluation_coco.py

clean:
	rm -f deconv/logs/*
	rm -f deconv/models/*