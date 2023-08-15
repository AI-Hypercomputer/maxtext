import datetime
from airflow import models
from apis import gcp_config, metric_config, test_config
from apis import task


# TODO(wcromar): Unify these with other tests
TPU_PROD_ENV_ONE_VM = "tpu-prod-env-one-vm"
EUROPE_WEST4_A = gcp_config.GCPConfig(
    TPU_PROD_ENV_ONE_VM,
    "europe-west4-a",
    metric_config.DatasetOption.XLML_DATASET,
)
US_CENTRAL2_B = gcp_config.GCPConfig(
    TPU_PROD_ENV_ONE_VM,
    "us-central2-b",
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id="pytorchxla-torchvision",
    schedule=None,
    tags=["pytorchxla", "latest", "supported"],
    start_date=datetime.datetime(2023, 7, 12),
):
  mnist_v2_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-mnist-pjrt-func-v2-8-1vm"
      ),
      EUROPE_WEST4_A,
  ).run()
  resnet_v2_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v2-8-1vm"
      ),
      EUROPE_WEST4_A,
  ).run()
  resnet_v4_8 = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  ).run()

  mnist_v2_8 >> resnet_v2_8
  mnist_v2_8 >> resnet_v4_8
