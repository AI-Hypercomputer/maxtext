FROM maxtext_base_image

RUN pip install torch

COPY . /maxtext

ENTRYPOINT ["bash"]
