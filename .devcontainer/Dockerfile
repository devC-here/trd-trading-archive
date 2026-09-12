# ==============================================================================
# TRD MetaTrader 5 on Codespaces Linux (Wine 9/10 + noVNC Web Desktop)
# ==============================================================================
FROM mcr.microsoft.com/devcontainers/base:jammy

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV WINEPREFIX=/home/vscode/.wine
ENV WINEARCH=win64
ENV WINEDEBUG=-all
ENV DISPLAY=:1
ENV WINEDLLOVERRIDES="mscoree,mshtml="

# 1. Enable 32-bit architecture & install core system dependencies
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        gnupg2 \
        software-properties-common \
        xvfb \
        x11vnc \
        x11-utils \
        fluxbox \
        novnc \
        websockify \
        cabextract \
        unzip \
        tar \
        git \
        procps \
        net-tools \
        python3 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install modern Wine from WineHQ (with fallback to distribution packages)
RUN mkdir -pm755 /etc/apt/keyrings && \
    wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key && \
    wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources && \
    apt-get update && \
    (apt-get install --install-recommends -y winehq-stable || apt-get install -y wine64 wine32 wine) && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Setup user environment and Wine prefix as 'vscode'
USER vscode
WORKDIR /home/vscode

# Initialize Wine in Windows 10 mode with headless Xvfb
RUN Xvfb :99 -screen 0 1024x768x16 & \
    XVFB_PID=$! && \
    sleep 2 && \
    DISPLAY=:99 wineboot -i && \
    DISPLAY=:99 wine reg add "HKCU\\Software\\Wine" /v Version /t REG_SZ /d "win10" /f && \
    kill $XVFB_PID 2>/dev/null || true

# 4. Install 64-bit Windows Python inside Wine for native MetaTrader5 Python library
RUN wget -q -O /tmp/python-3.9.13-amd64.exe https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe && \
    Xvfb :99 -screen 0 1024x768x16 & \
    XVFB_PID=$! && \
    sleep 2 && \
    DISPLAY=:99 wine /tmp/python-3.9.13-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 TargetDir="C:\\Python39" SimpleInstall=1 && \
    sleep 3 && \
    DISPLAY=:99 wine "C:\\Python39\\python.exe" -m pip install --upgrade pip && \
    DISPLAY=:99 wine "C:\\Python39\\python.exe" -m pip install --no-cache-dir MetaTrader5 pandas numpy && \
    rm -f /tmp/python-3.9.13-amd64.exe && \
    kill $XVFB_PID 2>/dev/null || true

# 5. Download official MetaTrader 5 installer
RUN curl -sSL "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -o /home/vscode/mt5setup.exe && \
    chmod +x /home/vscode/mt5setup.exe

# 6. Copy entrypoint and helper scripts
USER root
COPY entrypoint.sh /entrypoint.sh
COPY launch_mt5.sh /usr/local/bin/launch_mt5
RUN chmod +x /entrypoint.sh /usr/local/bin/launch_mt5 && \
    chown vscode:vscode /entrypoint.sh /usr/local/bin/launch_mt5

USER vscode

EXPOSE 6080 5901 5000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
