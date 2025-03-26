import subprocess

def get_gpu_info():
    try:
        # Run the command and capture the output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Print the output
        print(result.stdout)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

def main():
    f = open("test_output.txt", "w")
    f.write("Test completed.")
    f.close()
    get_gpu_info()

if __name__ == "__main__":
    main()