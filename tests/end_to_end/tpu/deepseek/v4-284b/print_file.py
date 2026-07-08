import sys

def main():
  filename = sys.argv[1] if len(sys.argv) > 1 else "conversion.log"
  print(f"--- CONTENT OF {filename} ---")
  try:
    with open(filename, "r") as f:
      print(f.read())
  except Exception as e:
    print(f"Error reading {filename}: {e}")
  print("-----------------------------")

if __name__ == "__main__":
  main()
