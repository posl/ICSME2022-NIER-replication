FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        sudo \
        wget \
        vim \
        git

WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
        sh Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
        rm -f Anaconda3-2019.10-Linux-x86_64.sh

ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip && pip install \
        pyqt5==5.12 \
        PyQtWebEngine==5.12 \
        Keras==2.2.4 \
        tensorflow==1.13.1 \
        cleverhans==3.0.1
RUN conda install lightgbm==3.2.1 -y
RUN conda install shap==0.39.0 -y

WORKDIR /work
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''", "--port=9999"]
