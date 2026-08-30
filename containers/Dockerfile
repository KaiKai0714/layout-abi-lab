ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/layout-abi-lab
COPY . /opt/layout-abi-lab
RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "layoutabi.cli"]
CMD ["--help"]

