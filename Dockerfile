FROM python:2.7

RUN git clone https://github.com/networkdynamics/zenlib /tmp/zenlib
RUN pip install --upgrade pip
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install cython
RUN pip install numpy
RUN pip install pandas
RUN pip install scipy
RUN pip install networkx
RUN cd /tmp/zenlib/src; python setup.py install
 
