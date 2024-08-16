# FROM --platform=linux/amd64 pytorch/pytorch
FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=user:user requirements.txt /opt/app/

# RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
# RUN python -m piptools sync requirements.txt

RUN python -m pip install --user torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user setup.py /opt/app/
COPY --chown=user:user inference_code.py /opt/app/
COPY --chown=user:user nnunet/ /opt/app/nnunet/

#only for testing postpro alg
# COPY --chown=user:user result/ /opt/app/result/
# COPY --chown=user:user input/ /input/

RUN python -m pip install --user -e .

# RUN python -m pip install \
#     --user \
#     --no-cache-dir \
#     --no-color \
#     --requirement /opt/app/requirements.txt

# COPY --chown=user:user inference.py /opt/app/

ENV nnUNet_raw_data_base="/opt/app/data/nnUNet_raw"
ENV nnUNet_preprocessedv1="/opt/app/data/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/opt/app/data/nnUNet_results"

ENTRYPOINT ["python", "inference.py"]
