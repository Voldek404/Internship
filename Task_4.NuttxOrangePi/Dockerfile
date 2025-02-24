FROM debian:bullseye

# Обновление списка пакетов
RUN apt-get update && apt-get upgrade -y 

# Установка необходимых пакетов для сборки
RUN apt-get install -y --no-install-recommends \
    gcc gcc-aarch64-linux-gnu bc bison build-essential coccinelle \
    device-tree-compiler dfu-util efitools flex gdisk graphviz imagemagick \
    liblz4-tool libgnutls28-dev libguestfs-tools libncurses-dev \
    libpython3-dev libsdl2-dev libssl-dev lz4 lzma openssl \
    pkg-config python3 python3-asteval python3-coverage python3-filelock \
    python3-pkg-resources python3-pycryptodome python3-pyelftools \
    python3-pytest python3-pytest-xdist python3-sphinxcontrib.apidoc \
    python3-sphinx-rtd-theme python3-subunit python3-testtools \
    python3-virtualenv swig uuid-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка дополнительных Python зависимостей
RUN pip3 install --upgrade pip
RUN pip3 install -r /u-boot/requirements.txt || echo "requirements.txt not found or error during installation."

# Копирование файлов проекта
COPY . /u-boot/

# Установка рабочего каталога
WORKDIR /u-boot

# Просмотр содержимого директории
RUN ls -l

# Сборка для платформы Orange Pi RK3399
RUN make rk3399_defconfig
RUN make -j$(nproc)
