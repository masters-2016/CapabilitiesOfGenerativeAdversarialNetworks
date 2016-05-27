FROM kaixhin/cuda-torch:7.0

#ADD cudnn /cudnn
#RUN cp /cudnn/include/* /usr/local/cuda/include/
#RUN cp /cudnn/lib64/* /usr/local/cuda/lib64/

#RUN luarocks install cudnn
#RUN luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
#RUN luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec

RUN mkdir /data
RUN mkdir /src
WORKDIR /src

CMD ["bash"]
