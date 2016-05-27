default: run

run: stop
	docker run -it \
		--name torch \
		-v $(shell pwd)/src:/src \
		-v $(shell pwd)/data:/data \
		-v $(shell pwd)/out:/out \
		torch

run-gpu: stop
	docker run -it \
		--name torch \
		-v $(shell pwd)/src:/src \
		-v $(shell pwd)/data:/data \
		-v $(shell pwd)/out:/out \
		--device /dev/nvidia0:/dev/nvidia0 \
		--device /dev/nvidiactl:/dev/nvidiactl \
		--device /dev/nvidia-uvm:/dev/nvidia-uvm \
		torch

stop:
	-docker stop torch
	-docker rm -v torch

build:
	docker build -t torch .

clean: stop
	docker rmi torch
