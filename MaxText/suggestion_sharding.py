from absl import app
from absl import flags
import argparse
from typing import Sequence
import numpy as np
from accelerator_to_spec_map import get_system_characteristics

parser = argparse.ArgumentParser(
    description="Return a guideline for sharding specification"
)
parser.add_argument(
    "--accelerator_type",
    "-at",
    required=True,
    default="",
    help="Accelerator type such as v5p, v5e, v6e",
    type=str,
)
parser.add_argument(
    "--accelerator",
    "-a",
    required=True,
    default="",
    help="Accelerator such as v5e-128",
    type=str,
)
parser.add_argument(
    "--num_slices",
    "-ns",
    required=False,
    default=1,
    help="Number of slices",
    type=int,
)
parser.add_argument(
    "--global_batch",
    "-b",
    required=False,
    default=2**23, #8M
    help="Global batch size",
    type=int,
)
parser.add_argument(
    "--per_device_batch",
    "-pdb",
    required=False,
    help="per device batch size",
    type=int,
)
parser.add_argument(
    "--token_factor",
    "-tf",
    required=False,
    default=20,
    help="Ratio of tokens trained for convergence to model size",
    type=int,
)
parser.add_argument(
    "--ai_breathing_factor",
    "-abf",
    required=False,
    default=2.0,
    help="Desired factor of model AI to hardware AI, should be at least 1.0",
    type=float,
)
parser.add_argument(
    "--embed",
    "-emb",
    required=False,
    default=2048,
    help="Embed dimension",
    type=float,
)
parser.add_argument(
    "--mlp",
    "-mlp",
    required=False,
    default=8192,
    help="MLP dim",
    type=int,
)
parser.add_argument(
    "--experts",
    "-exp",
    required=False,
    default=1,
    help="Number of experts",
    type=int,
)
parser.add_argument(
    "--layers",
    "-l",
    required=False,
    default=16,
    help="Number of layers",
    type=int,
)
parser.add_argument(
    "--seq_length",
    "-seq",
    required=False,
    default=2048,
    help="Sequence Length",
    type=int,
)
parser.add_argument(
    "--fsdp_parallelism",
    "-fsdp",
    required=False,
    default=1,
    help="fsdp parallelism",
    type=int,
)
parser.add_argument(
    "--tp_parallelism",
    "-tp",
    required=False,
    default=1,
    help="tp parallelism",
    type=int,
)
parser.add_argument(
    "--pipeline_parallelism",
    "-pp",
    required=False,
    default=1,
    help="pipeline_parallelism",
    type=int,
)

global args
args = parser.parse_args()

SECONDS_IN_30_DAYS = 60 * 60 * 24 * 30
SECONDS_IN_60_DAYS = 60 * 60 * 24 * 60

def calculate_fsdp_boundary():
  # model_arithmetic_intensity = batch axis len = global_batch / chips
  # chips is calculated to train in a certain number of days -
  # Total flops to train to convergence is is 6 * Params * Tokens = 6 * Params^2 * token_factor
  # Thus to train to 30 days, it will take 6 * Params^2 * token_factor / (chip_speed * SECONDS_IN_30_DAYS) chips
  # So the model arithmetic intensity is G * C * 30d / (6 * P^2 * tf)
  # The hardware arithmetic intensity is C / ag_speed
  # If we want model arithmetic intensity = hardware arithmetic intensity * ai_breathing_factor, and solve for params,
  # we get the below equation
  fsdp_boundary_params = np.sqrt(args.global_batch * args.ag_speed * SECONDS_IN_30_DAYS / (6 * args.token_factor * args.ai_breathing_factor))
  return fsdp_boundary_params

def calculate_fsdp_ag_ai_ratio():
    assert args.sharding["fsdp"] > 1
    def calculate_fsdp_ag_model_intensity():
        return args.global_batch // args.sharding["fsdp"]
    def calculate_fsdp_ag_hardware_intensity():
        return args.chip_speed / (args.ici_speed * args.num_ici["fsdp"])
    model_ai = calculate_fsdp_ag_model_intensity()
    hardware_ai = calculate_fsdp_ag_hardware_intensity()
    ai_ratio = model_ai / hardware_ai
    print(f"AG summary: Batch {args.global_batch} over {args.sharding['fsdp']} FSDP Shards using {args.num_ici['fsdp']} ici links")
    print(f"AG Model AI: {model_ai:0.2g}, hardawre AI {hardware_ai:0.2g}, ratio {ai_ratio:0.2g}")
    return ai_ratio

def calculate_dcn_ar_ai_ratio():
   # Focuses on the slice scale - the compute and communication of an entire slice (Assuming mix of DP and PP across slices)
   # Alternatively can look at chip scale - compute and communication are both scaled down by a factor of chips per slice
   assert args.sharding['dp_dcn'] > 1
   batch_per_dcn_data_replica = args.global_batch // args.sharding['dp_dcn'] 
   def calculate_dcn_ar_model_ai():
      # Compute: 4 * Batch * Params Flops in backward pass
      # Communication: 2 Bytes/param * Params 
      # Note with PP Params doesn't refer to all model params but only that stage's responsibility -
      # e.g. Params = All Params / PP shards, but this # PP shards appears in both compute and communication.
      # PP is still useful because the batch is less sharded with PP - e.g. instead of
      # batch_per_slice = global_batch / num_slices, it is instead
      # batch_per_slice = global_batch / dp_dcn_parallelism
      return 2 * batch_per_dcn_data_replica
   def calculate_dcn_ar_hardware_ai():
      # We see that slice_size cancels below. The reason having a larger slice size is helpful is actually
      # hidden in the model AI above - with fewer slices we have a smaller dp_dcn_parallelism so a larger model AI
      return args.chip_speed * args.slice_size /  args.dcn_speed  * args.slice_size
   
   model_ai = calculate_dcn_ar_model_ai()
   hardware_ai = calculate_dcn_ar_hardware_ai()
   ai_ratio = model_ai / hardware_ai
   print(f"DCN reduce summary: Global batch {args.global_batch} over {args.sharding['dp_dcn']} DP shards for per-slice of {batch_per_dcn_data_replica}")
   print(f"DCN reduce Model AI: {model_ai:0.2g}, hardawre AI {hardware_ai:0.2g}, ratio {ai_ratio:0.2g}")
   return ai_ratio

def calculate_tp_ai_ratio():
    assert args.sharding["tp"] > 1
    def calculate_tp_model_intensity():
        return np.min((args.embed, args.mlp)) // args.sharding["tp"]
    def calculate_tp_hardware_intensity():
        return args.chip_speed / (args.ici_speed * args.num_ici["tp"])
    model_ai = calculate_tp_model_intensity()
    hardware_ai = calculate_tp_hardware_intensity()
    ai_ratio = model_ai / hardware_ai
    print(f"TP summary: Embed dim {args.embed} over {args.sharding['tp']} TP Shards using {args.num_ici['tp']} ici links")
    print(f"TP Model AI: {model_ai:0.2g}, hardawre AI {hardware_ai:0.2g}, ratio {ai_ratio:0.2g}")
    return ai_ratio

def hardware_specs(accelerator_type):
  if accelerator_type == 'v5p':
    args.chip_speed = 459 * 10**12
    args.ici_speed = 0.1 * 10**12
    args.ag_speed = 0.6 * 10**12 
    args.dcn_speed = 12.5 * 10**9
    args.slice_size = 2048
  else:
    raise ValueError(f"unsupported accelerator type {accelerator_type}")
def parse_specs_to_mesh(acc_name):
   a = acc_name.split(":")[1].split("x")
   return [int(m) for m in a]

def init_shardings():
   keys = ["fsdp", "tp", "pp", "dp_dcn", "pp_dcn"]
   args.sharding = {key: 1 for key in keys}
   args.num_ici = {key: 0 for key in keys} 

def sharding_to_parallelism(mesh, sharding):
   args.sharding = {key: 1 for key in args.sharding}  
   args.num_ici = {key: 0 for key in args.num_ici} 
   for dim, parallelism_type in zip(mesh, sharding):
      args.sharding[parallelism_type] =  dim * args.sharding[parallelism_type]
      args.num_ici[parallelism_type] = args.num_ici[parallelism_type] + 2

def full_fsdp():
   sharding = ['fsdp', 'fsdp', 'fsdp']
   sharding_to_parallelism(args.mesh, sharding)
   print("\n === Using pure FSDP === \n")
   ai_ratio = calculate_fsdp_ag_ai_ratio()

def fsdp_and_tp():
   sharding = ['fsdp', 'fsdp', 'tp']
   sharding_to_parallelism(args.mesh, sharding)
   print("\n === Using FSDP and TP mix === \n")
   fsdp_ai_ratio = calculate_fsdp_ag_ai_ratio()
   tp_ai_ratio = calculate_tp_ai_ratio()

def fsdp_and_pp():
   sharding = ['fsdp', 'fsdp', 'pp']
   sharding_to_parallelism(args.mesh, sharding)
   print("\n === Using FSDP and PP mix === \n")
   ai_ratio = calculate_fsdp_ag_ai_ratio()
   

def main(_argv: Sequence[str]) -> None:
  hardware_specs(args.accelerator_type)
  hardware_characteristics = get_system_characteristics(args.accelerator)
  init_shardings()
  args.mesh = parse_specs_to_mesh(hardware_characteristics.topology_name)
  full_fsdp()
  fsdp_and_tp()



def parse_flags(argv):
  return parser.parse_args(argv[1:])


if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  app.run(main, flags_parser=parse_flags)