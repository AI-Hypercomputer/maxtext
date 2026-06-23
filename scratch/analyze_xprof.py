import sys
from tensorflow.tsl.profiler.protobuf import xplane_pb2
from google.protobuf import text_format

def analyze_xplane(file_path):
  print(f"Analyzing {file_path}...")
  xspace = xplane_pb2.XSpace()
  with open(file_path, "rb") as f:
    xspace.ParseFromString(f.read())

  kernel_events = []
  
  for plane in xspace.planes:
    # Build metadata map
    metadata_map = {}
    for event_id, event_meta in plane.event_metadata.items():
      metadata_map[event_id] = event_meta.name
      
    for line in plane.lines:
      for event in line.events:
        name = metadata_map.get(event.metadata_id, "")
        if "sc_ragged_gather_reduce" in name:
          duration_ns = event.duration_ps / 1000.0
          kernel_events.append((name, duration_ns))
          
  if not kernel_events:
    print("No sc_ragged_gather_reduce events found!")
    return
    
  print(f"Found {len(kernel_events)} events:")
  # Sort by duration descending
  kernel_events.sort(key=lambda x: x[1], reverse=True)
  for name, duration_ns in kernel_events[:20]:
    print(f"  {name}: {duration_ns:,.2f} ns ({duration_ns/1000.0:,.2f} us)")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 analyze_xprof.py <path_to_xplane.pb>")
    sys.exit(1)
  analyze_xplane(sys.argv[1])
