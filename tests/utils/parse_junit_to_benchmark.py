import xml.etree.ElementTree as ET
import glob
import json
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_junit_to_benchmark.py <xml_dir> <output_json>")
        sys.exit(1)
        
    xml_dir = sys.argv[1]
    output_json = sys.argv[2]
    
    benchmarks = []
    total_times_by_device = {}
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in xml_files:
        basename = os.path.basename(xml_file)
        # e.g., test-results-tpu-1.xml -> device = tpu
        device = "unknown"
        parts = basename.replace(".xml", "").split("-")
        if len(parts) >= 3:
            device = parts[2]
            
        try:
            tree = ET.parse(xml_file)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
            
        root = tree.getroot()
        
        for testsuite in root.iter('testsuite'):
            for testcase in testsuite.iter('testcase'):
                name = testcase.get('name')
                classname = testcase.get('classname')
                time_val = float(testcase.get('time', 0.0))
                
                # Prefix with device to distinguish test times on different hardware
                full_name = f"[{device.upper()}] {classname}::{name}"
                
                benchmarks.append({
                    "name": full_name,
                    "unit": "sec",
                    "value": time_val
                })
                
                total_times_by_device[device] = total_times_by_device.get(device, 0.0) + time_val

    for device, total_time in total_times_by_device.items():
        benchmarks.append({
            "name": f"Total {device.upper()} Test Suite Time",
            "unit": "sec",
            "value": total_time
        })
        
    with open(output_json, "w") as f:
        json.dump(benchmarks, f, indent=2)
        
    print(f"Parsed {len(xml_files)} XML files and extracted {len(benchmarks)} duration metrics.")

if __name__ == "__main__":
    main()
