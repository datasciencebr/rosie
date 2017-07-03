FROM python:3.5-alpine

WORKDIR /home

COPY requirements.txt ./

ADD http://www.netlib.org/blas/blas-3.5.0.tgz /tmp/blas.tgz
ADD http://www.netlib.org/lapack/lapack-3.5.0.tgz /tmp/lapack.tgz

    # install basic build dependencies
RUN apk update && apk upgrade && apk add --no-cache --virtual=build_dependencies \
        cmake \
        g++ \
        gcc \
        gfortran \
        git \
        libffi \
        libxml2 \
        libxml2-dev \
        libxslt \
        libxslt-dev \
        make \
        musl-dev && \

    # fix numpy installing error
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \

    # compile BLAS (scipy dependency)
    mkdir -p ~/src/ && \
    cd ~/src/ && \
    mv /tmp/blas.tgz . && \
    tar xzf blas.tgz && \
    cd BLAS-3.5.0 && \
    gfortran -O3 -std=legacy -m64 -fno-second-underscore -fPIC -c *.f && \
    ar r libfblas.a *.o && \
    ranlib libfblas.a && \
    rm -rf *.o && \

    # compile LAPACK (scipy dependency)
    mkdir -p ~/src/ && \
    cd ~/src/ && \
    mv /tmp/lapack.tgz . && \
    tar xzf lapack.tgz && \
    cd lapack-3.5.0/ && \
    cp make.inc.example make.inc && \
    sed -i -- 's/frecursive/fPIC/g' make.inc && \
    make lapacklib -j$(cat /proc/cpuinfo | grep 'processor' | wc -l) && \
    make clean && \
    cd /home && \

    pip install numpy && \

    BLAS=~/src/BLAS-3.5.0/libfblas.a LAPACK=~/src/lapack-3.5.0/liblapack.a pip install $(grep "scipy" requirements.txt) && \

    pip install -r requirements.txt && \

    # cleanup
    apk del build_dependencies && \
    apk add --no-cache libstdc++ libgfortran libxslt && \
    rm -rf /var/cache/apk/* && \
    rm -rf ~/src


COPY rosie ./rosie
COPY rosie.py config.ini ./

VOLUME /tmp/serenata-data

CMD python rosie.py run
